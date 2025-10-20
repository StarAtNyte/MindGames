import modal
import re
import time
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Modal app
app = modal.App("enhanced-secretmafia")

# Create volume for fine-tuned model (same as training volume)
finetuned_volume = modal.Volume.from_name("mafia-data", create_if_missing=True)

# Model selection - change this variable to switch models
USE_FINETUNED_MODEL = False  # Set to True for fine-tuned, False for base model

# Define the image with required dependencies
image = (
    modal.Image.debian_slim()
    .apt_install([
        "git"
    ])
    .pip_install([
        "transformers>=4.51.0",
        "torch>=2.0.0",
        "accelerate>=0.21.0",
        "bitsandbytes>=0.41.0",
        "fastapi>=0.104.0",
        "psutil>=5.9.0",
        "hf_transfer",
        "tiktoken"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Request model
class GenerateRequest(BaseModel):
    observation: str
    tom_insights: Optional[str] = ""
    strategic_context: Optional[str] = ""
    system_prompt: Optional[str] = ""
    phase: Optional[str] = "discussion"

class GenerateResponse(BaseModel):
    response: str
    error: Optional[str] = None
    error_type: Optional[str] = None
    processing_time: Optional[float] = None
    reasoning: Optional[str] = None  # Added for transparency

@app.cls(
    image=image,
    gpu="A100",
    memory=32768,
    timeout=1200,
    scaledown_window=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/finetuned": finetuned_volume},  # Mount volume for fine-tuned model
)
@modal.concurrent(max_inputs=10)
class EnhancedSecretMafiaModel:
    model_name: str = modal.parameter(default="Qwen/Qwen3-8B")
    
    @modal.enter()
    def load_model(self):
        """Load the model with enhanced social deduction capabilities"""
        start_time = time.time()
        logger.info("=== ENHANCED SECRETMAFIA MODEL INITIALIZATION ===")
        
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import os
        
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        try:
            # Determine model path
            if USE_FINETUNED_MODEL:
                finetuned_path = "/finetuned/qwen3-mafia-merged"
                if os.path.exists(finetuned_path):
                    model_path = finetuned_path
                    logger.info(f"🎯 Using fine-tuned model: {model_path}")
                else:
                    logger.warning(f"Fine-tuned model not found at {finetuned_path}, falling back to base model")
                    model_path = self.model_name
            else:
                model_path = self.model_name
                logger.info(f"🔤 Using base model: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path if not model_path.startswith('/') else self.model_name,  # Use base for tokenizer if local path
                trust_remote_code=True,
                token=hf_token
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True,
                low_cpu_mem_usage=True,
                token=hf_token if not model_path.startswith('/') else None  # No token needed for local path
            )
            
            # Create pipeline
            self.pipe = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                do_sample=True,
                temperature=0.7,
                top_k=20,
                top_p=0.8,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            total_time = time.time() - start_time
            model_type = "Fine-tuned" if USE_FINETUNED_MODEL and model_path.startswith('/') else "Base"
            logger.info(f"Enhanced {model_type} model loaded in {total_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")
            raise
    
    def _extract_game_info(self, observation: str) -> Dict:
        """Extract key game information from observation"""
        info = {
            'phase': None,
            'role': None,
            'player_id': None,
            'alive_players': [],
            'day_number': 1,
            'valid_targets': []
        }
        
        # Extract phase - FIXED detection logic to properly identify discussion phases
        phase_indicators = {
            # Discussion phases (check FIRST - highest priority)
            'Day breaks': 'DAY_DISCUSSION',
            'Discuss for': 'DAY_DISCUSSION',
            'Round 1 Discussion': 'DAY_DISCUSSION',
            'Round 2 Discussion': 'DAY_DISCUSSION', 
            'Round 3 Discussion': 'DAY_DISCUSSION',
            'Discussion Phase': 'DAY_DISCUSSION',
            'What do you say': 'DAY_DISCUSSION',
            'What do you think': 'DAY_DISCUSSION',
            
            # Night phases
            'Night has fallen': 'NIGHT_MAFIA',
            'Mafia, agree on a victim': 'NIGHT_MAFIA',
            'choose one player to protect': 'NIGHT_DOCTOR',
            'choose one player to investigate': 'NIGHT_DETECTIVE',
            
            # Voting phases (check LAST - lowest priority)
            'Voting phase': 'DAY_VOTING',
            'submit one vote in format': 'DAY_VOTING'
        }
        
        detected_phase = None
        
        # CRITICAL FIX: Check for discussion indicators first
        # If we see discussion content, it's a discussion phase even if voting is mentioned
        discussion_content_indicators = [
            'Day breaks', 'Discuss for', 'Round 1 Discussion', 'Round 2 Discussion', 'Round 3 Discussion'
        ]
        
        has_discussion_content = any(indicator in observation for indicator in discussion_content_indicators)
        
        # Check for player statements (strong indicator of discussion phase)
        has_player_statements = bool(re.search(r'\[Player \d+\] \[.*?\]', observation))
        
        if has_discussion_content or has_player_statements:
            detected_phase = 'DAY_DISCUSSION'
            logger.info(f"DISCUSSION PHASE DETECTED - Content: {has_discussion_content}, Statements: {has_player_statements}")
        else:
            # Only then check other phase indicators
            for indicator, phase in phase_indicators.items():
                if indicator in observation:
                    detected_phase = phase
                    logger.info(f"PHASE DETECTED: {phase} from indicator: '{indicator}'")
                    break
        
        # Role-aware phase correction to prevent mismatches
        if detected_phase and info['role']:
            role = info['role'].lower()
            # Villagers have no night actions - if it's a night action phase, make it discussion
            if role == 'villager' and detected_phase.startswith('NIGHT_'):
                if 'Day breaks' in observation or 'Discuss' in observation:
                    info['phase'] = 'DAY_DISCUSSION'
                else:
                    info['phase'] = 'NIGHT_WAIT'  # Villager waiting during night
            else:
                info['phase'] = detected_phase
        else:
            info['phase'] = detected_phase
        
        # Final fallback: if no explicit phase found, check for any player content
        if info['phase'] is None:
            if 'Player' in observation:
                info['phase'] = 'DAY_DISCUSSION'
                logger.info("FALLBACK: Detected DAY_DISCUSSION from Player mentions")
        
        # Extract role
        role_match = re.search(r'Your role: (\w+)', observation)
        if role_match:
            info['role'] = role_match.group(1)
        
        # Extract player ID
        player_match = re.search(r'You are Player (\d+)', observation)
        if player_match:
            info['player_id'] = int(player_match.group(1))
        
        # Extract valid targets - handle both space and comma separated formats
        # Format 1: "Valid targets: [0], [1], [2], [3]" (comma-separated - ACTUAL format for night phase)
        # Format 2: "choose one player to investigate: [0], [1], [2], [3]" (comma-separated)
        # Format 3: "Valid: [0], [1], [2], [3]" (comma-separated - voting)  
        valid_target_patterns = [
            r'Valid targets:\s*((?:\[\d+\](?:,\s*)?)+)',  # "Valid targets: [0], [1], [2]" (night phase) - Fixed pattern
            r'Valid:\s*((?:\[\d+\](?:,\s*)?)+)',  # "Valid: [0], [1], [2]" (voting) - Fixed pattern
            r'choose one player to (?:protect|investigate|eliminate):\s*((?:\[\d+\](?:,\s*)?)+)',  # "choose...: [0], [1], [2]" - Fixed pattern
            r'submit one vote in format \[X\]\. Valid:\s*((?:\[\d+\](?:,\s*)?)+)',  # Full voting format - Fixed pattern
        ]
        
        valid_targets = []
        for i, pattern in enumerate(valid_target_patterns):
            match = re.search(pattern, observation, re.IGNORECASE | re.DOTALL)
            if match:
                # Extract all numbers from the matched group (handles both comma and space separated)
                target_matches = re.findall(r'(\d+)', match.group(1))
                valid_targets = [int(t) for t in target_matches]
                logger.info(f"Pattern {i+1} matched: '{match.group(0)}' -> extracted targets: {valid_targets}")
                break
            else:
                logger.debug(f"Pattern {i+1} '{pattern}' did not match")
        
        # Enhanced fallback: look for lines with multiple bracket patterns
        # Also run if we got suspiciously few targets (likely partial match)
        if not valid_targets or (len(valid_targets) == 1 and info['alive_players'] and len(info['alive_players']) > 3):
            if valid_targets:
                logger.warning(f"Suspiciously few targets ({valid_targets}) for {len(info['alive_players'])} alive players - trying fallback")
            
            logger.info(f"Running fallback target extraction on observation: {repr(observation)}")
            
            # Try a more aggressive approach - find all [X] patterns in the entire observation
            all_bracket_targets = re.findall(r'\[(\d+)\]', observation)
            logger.info(f"All bracket patterns found: {all_bracket_targets}")
            
            # Look for lines that contain target information
            lines = observation.split('\n')
            fallback_targets = []
            
            for line in lines:
                logger.debug(f"Checking line: {repr(line)}")
                # Look for lines with multiple [X] patterns indicating valid targets
                if any(keyword in line.lower() for keyword in ['valid', 'choose', 'protect', 'investigate', 'vote', 'target']):
                    target_matches = re.findall(r'\[(\d+)\]', line)
                    logger.info(f"Line with keywords found: {repr(line)} -> targets: {target_matches}")
                    if len(target_matches) >= 2:  # Multiple targets indicate valid options
                        fallback_targets = [int(t) for t in target_matches]
                        logger.info(f"Fallback extracted valid targets from line: '{line}' -> {fallback_targets}")
                        break
                    elif len(target_matches) == 1 and not fallback_targets:
                        # Store single target as potential fallback
                        fallback_targets = [int(t) for t in target_matches]
                        logger.info(f"Fallback extracted single target from line: '{line}' -> {fallback_targets}")
            
            # If still no targets, try to extract from context clues
            if not fallback_targets and all_bracket_targets:
                # Filter out likely player IDs (self and teammates)
                player_id = info.get('player_id', -1)
                potential_targets = []
                for target_str in all_bracket_targets:
                    target = int(target_str)
                    # Exclude self and known teammates (for Mafia)
                    if target != player_id:
                        potential_targets.append(target)
                
                if potential_targets:
                    fallback_targets = potential_targets
                    logger.info(f"Using context-based fallback targets: {fallback_targets}")
            
            # Use fallback targets if they're more comprehensive
            if len(fallback_targets) > len(valid_targets):
                logger.info(f"Using fallback targets {fallback_targets} instead of original {valid_targets}")
                valid_targets = fallback_targets
        
        # Emergency fallback: if we still have no targets but we know it's an action phase, 
        # generate targets from alive players (excluding self for Mafia night actions)
        if not valid_targets and info['phase'] in ['NIGHT_MAFIA', 'NIGHT_DOCTOR', 'NIGHT_DETECTIVE', 'DAY_VOTING']:
            logger.warning("Emergency fallback: generating targets from alive players")
            player_id = info.get('player_id', -1)
            
            # For night phases, exclude self and teammates
            if info['phase'] == 'NIGHT_MAFIA' and info['role'] == 'Mafia':
                # Exclude self and other Mafia members (teammates)
                emergency_targets = []
                for p in info['alive_players']:
                    if p != player_id:  # Exclude self
                        emergency_targets.append(p)
                # Remove known teammates if we can identify them from observation
                teammate_matches = re.findall(r'Your teammates are: ([^.]+)', observation)
                if teammate_matches:
                    teammate_text = teammate_matches[0]
                    teammate_ids = re.findall(r'Player (\d+)', teammate_text)
                    for teammate_str in teammate_ids:
                        teammate_id = int(teammate_str)
                        if teammate_id in emergency_targets:
                            emergency_targets.remove(teammate_id)
                valid_targets = emergency_targets
                logger.warning(f"Emergency Mafia targets generated: {valid_targets}")
            
            elif info['phase'] in ['NIGHT_DOCTOR', 'NIGHT_DETECTIVE']:
                # For Doctor/Detective, can target anyone alive (including self for Doctor)
                valid_targets = info['alive_players'].copy()
                logger.warning(f"Emergency {info['role']} targets generated: {valid_targets}")
            
            elif info['phase'] == 'DAY_VOTING':
                # For voting, exclude self
                valid_targets = [p for p in info['alive_players'] if p != player_id]
                logger.warning(f"Emergency voting targets generated: {valid_targets}")

        # Debug logging for target extraction
        if valid_targets:
            logger.info(f"Final valid targets: {valid_targets} (from {len(info['alive_players'])} alive players)")
        else:
            logger.error(f"FAILED to extract any valid targets from observation: {observation}")
            # Show relevant lines for debugging
            lines = observation.split('\n')
            relevant_lines = [line for line in lines if any(keyword in line.lower() for keyword in ['valid', 'choose', 'protect', 'investigate', 'vote'])]
            logger.error(f"Relevant lines for debugging: {relevant_lines}")
        
        info['valid_targets'] = valid_targets
        
        info['valid_targets'] = valid_targets
        
        # Extract alive players
        alive_matches = re.findall(r'Player (\d+)', observation)
        info['alive_players'] = [int(p) for p in set(alive_matches)]
        
        return info
    
    def _parse_qwen_output(self, response: str) -> str:
        """Parse Qwen3-8B model output - handle thinking mode"""
        # Qwen3-8B with thinking mode generates <think>...</think> content
        # Extract the final response after thinking
        
        # Check if thinking mode was used (contains </think> marker)
        if '</think>' in response:
            try:
                # Split at </think> and take the content after
                parts = response.split('</think>')
                if len(parts) > 1:
                    # Get the response part (after thinking)
                    actual_response = parts[-1].strip()
                    if actual_response:
                        response = actual_response
                    else:
                        # If no content after </think>, use the thinking content
                        thinking_part = parts[0].replace('<think>', '').strip()
                        if thinking_part:
                            response = thinking_part
            except Exception as e:
                logger.warning(f"Error parsing thinking content: {e}")
        
        # Remove any remaining XML-style tags if they somehow appear
        response = re.sub(r'</?[^>]*>', '', response).strip()
        
        # Remove excessive whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # If empty after cleaning, return fallback
        if not response:
            return "I need to analyze the situation."
            
        return response
    
    def _generate_enhanced_prompt(self, observation: str, game_info: Dict, tom_insights: str = "", strategic_context: str = "") -> str:
        """Generate enhanced prompt with ToM insights and strategic context"""
        
        # Base system prompt with social deduction expertise
        system_prompt = """You are an expert social deduction player in Secret Mafia. You excel at:

STRATEGIC ANALYSIS:
- Reading behavioral patterns and inconsistencies
- Timing information reveals for maximum impact  
- Building trust while identifying deception
- Adapting communication style to the situation

ROLE-SPECIFIC TACTICS:
- Villager: Logical deduction, pattern analysis, trust-building
- Mafia: Misdirection, false accusations, blending in
- Doctor: Strategic protection, information gathering
- Detective: Careful investigation reveals, evidence presentation

COMMUNICATION PRINCIPLES:
- For discussions: Be detailed and analytical (20-60 words)
- For actions: Be concise (exact format required)
- Show reasoning: "I think X because Y"
- Reference specific game events and patterns
- Maintain consistent persona throughout
- Use strategic timing for maximum impact

RESPONSE FORMATS:
- Discussion: Natural conversational response (20-60 words) - NEVER use [number] format during discussion
- Voting: ONLY "[number]" - nothing else, no explanations
- Night actions: ONLY "[number]" - nothing else, no explanations, no "I think"
- CRITICAL: During discussion, respond with words and analysis, NOT numbers in brackets
- CRITICAL: During actions (voting/night), respond ONLY with [number], NO TEXT AT ALL
- NEVER include explanations with actions, just the number in brackets for actions only"""

        # Phase-specific guidance
        phase_guidance = ""
        if game_info['phase'] == 'DAY_DISCUSSION':
            phase_guidance = """
DISCUSSION STRATEGY:
- Provide detailed analysis (20-60 words) using natural language
- NEVER respond with [number] format during discussion
- Analyze who's been quiet vs active
- Reference voting patterns from previous rounds
- Look for defensive or aggressive behavior shifts
- Build logical cases with evidence
- Share your reasoning and observations
- Ask questions to gather more information
- Maintain your role's perspective consistently
- Example discussion response: "I think Player X is suspicious because they've been very quiet and their voting pattern seems inconsistent with their earlier statements."
- WRONG discussion format: "[3]" or "[Player 3]" - this is only for voting/actions!"""
            
        elif game_info['phase'] == 'DAY_VOTING':
            phase_guidance = """
VOTING STRATEGY:
- Consider behavioral patterns and inconsistencies
- Analyze who benefits from eliminations
- Factor in discussion contributions and timing
- Vote for most suspicious based on evidence
- Format: [number] only"""
            
        elif game_info['phase'] in ['NIGHT_MAFIA', 'NIGHT_DOCTOR', 'NIGHT_DETECTIVE']:
            # Filter out current player from valid targets for night actions
            filtered_targets = [t for t in game_info['valid_targets'] if t != game_info['player_id']]
            phase_guidance = f"""
NIGHT ACTION STRATEGY:
- Mafia: Target most trusted/least suspicious players
- Doctor: Protect likely town members (NOT yourself)  
- Detective: Investigate most suspicious players (NOT yourself)
- CRITICAL: You MUST respond with ONLY the target number in brackets: [X]
- DO NOT write any discussion or explanation text
- DO NOT write "I think" or analysis - just the action
- Valid targets: {filtered_targets} (you CANNOT target yourself - Player {game_info['player_id']})
- Example response: [{filtered_targets[0] if filtered_targets else 0}] (pick from valid targets only)
- WRONG: "I think Player 1 is suspicious. [1]" 
- CORRECT: "[{filtered_targets[0] if filtered_targets else 0}]" """
        
        elif game_info['phase'] == 'NIGHT_WAIT':
            phase_guidance = """
NIGHT WAITING:
- You have no night actions as a Villager
- Wait for day phase to begin
- Respond with simple acknowledgment: "Waiting for day."
- Do not attempt to take actions"""
        
        # Role-specific context
        role_context = ""
        if game_info['role'] == 'Mafia':
            role_context = """
MAFIA MINDSET:
- Appear helpful and analytical
- Subtly redirect suspicion to villagers
- Support other Mafia without being obvious
- Create doubt about detective claims
- Vote strategically to eliminate threats"""
            
        elif game_info['role'] in ['Villager', 'Doctor', 'Detective']:
            role_context = """
VILLAGE MINDSET:
- Focus on logical deduction
- Build trust through consistent behavior
- Share information strategically
- Identify behavioral inconsistencies
- Coordinate with other villagers"""
        
        # Add ToM insights and strategic context if provided
        enhancement_section = ""
        if tom_insights:
            enhancement_section += f"\nTHEORY OF MIND INSIGHTS:\n{tom_insights}\n"
        if strategic_context:
            enhancement_section += f"\nSTRATEGIC CONTEXT:\n{strategic_context}\n"

        # Combine all elements
        action_reminder = ""
        if game_info['phase'] in ['NIGHT_MAFIA', 'NIGHT_DOCTOR', 'NIGHT_DETECTIVE']:
            # Filter out current player for night actions
            filtered_targets = [t for t in game_info['valid_targets'] if t != game_info['player_id']]
            action_reminder = f"""
CRITICAL ACTION PHASE REMINDER:
- This is {game_info['phase']} - you MUST respond with ONLY [number] format
- Valid targets: {filtered_targets} (you CANNOT target yourself - Player {game_info['player_id']})
- DO NOT write any text, analysis, or explanation
- Example correct response: [{filtered_targets[0] if filtered_targets else 0}]
- Example WRONG response: "I think Player 1 is suspicious. [1]"
"""
        elif game_info['phase'] == 'DAY_VOTING':
            action_reminder = f"""
VOTING PHASE - CRITICAL FORMAT REQUIREMENT:
- This is {game_info['phase']} - you MUST respond with EXACTLY: [number]
- Valid targets: {game_info['valid_targets']}
- Choose ONE target to eliminate
- Response format: [X] where X is your chosen target
- Example: [2] or [4] (pick from valid targets only)
- DO NOT add any text, explanation, reasoning, or discussion
- WRONG: "I think Player 1 is suspicious. [1]"
- WRONG: "Based on evidence, [3]"
- CORRECT: "[2]"
"""
        elif game_info['phase'] == 'DAY_DISCUSSION':
            action_reminder = """
CRITICAL DISCUSSION PHASE REMINDER:
- This is DAY_DISCUSSION - you MUST respond with natural language, NOT [number] format
- Write 20-60 words of analysis and reasoning
- Example correct response: "I think Player 2 is acting suspiciously because they've been deflecting attention."
- Example WRONG response: "[2]"
"""

        # For voting phases, use ultra-simple prompt to ensure correct format
        if game_info['phase'] == 'DAY_VOTING':
            full_prompt = f"""You are voting in a Mafia game. You must eliminate one player.

VOTING INSTRUCTION:
- Respond with ONLY: [number]
- Valid targets: {game_info['valid_targets']}
- Choose the most suspicious player to eliminate
- Format: [X] where X is a number from the valid targets

Your vote:"""
        else:
            full_prompt = f"""{system_prompt}

{phase_guidance}

{role_context}{enhancement_section}

CURRENT GAME STATE:
{observation}

{action_reminder}

Analyze the situation carefully and provide your response directly (following the phase-specific format requirements above).

Your response:"""
        
        return full_prompt
    
    def _extract_move(self, response: str, game_info: Dict) -> str:
        """Enhanced move extraction with context awareness"""
        logger.debug(f"Extracting move from: '{response}'")
        
        response = response.strip()
        
        # For voting/action phases, prioritize bracket format  
        if game_info['phase'] in ['DAY_VOTING', 'NIGHT_MAFIA', 'NIGHT_DOCTOR', 'NIGHT_DETECTIVE']:
            logger.info(f"Processing action phase: {game_info['phase']}, valid targets: {game_info['valid_targets']}")
            
            # Check if model generated discussion text during action phase (error)
            if len(response.split()) > 5 and not re.search(r'\[(\d+)\]', response):
                logger.warning(f"MODEL ERROR: Generated discussion text '{response[:50]}...' during ACTION phase!")
            
            # Look for [number] format first
            match = re.search(r'\[(\d+)\]', response)
            if match:
                target = int(match.group(1))
                if target in game_info['valid_targets']:
                    result = f"[{target}]"
                    logger.info(f"Extracted valid bracket target: {result}")
                    return result
                else:
                    logger.warning(f"Bracket target {target} not in valid targets {game_info['valid_targets']}")
            
            # Look for standalone numbers in the response
            numbers = re.findall(r'\b(\d+)\b', response)
            for num_str in numbers:
                target = int(num_str)
                if target in game_info['valid_targets']:
                    result = f"[{target}]"
                    logger.info(f"Extracted valid number target: {result}")
                    return result
            
            # If response is very short or malformed, try to extract any number
            if len(response) < 20:  # Very short response, likely incomplete
                logger.warning(f"Very short response detected: '{response}'")
                # Look for any digit that might be a target
                digit_match = re.search(r'(\d)', response)
                if digit_match:
                    target = int(digit_match.group(1))
                    if target in game_info['valid_targets']:
                        result = f"[{target}]"
                        logger.info(f"Extracted digit from short response: {result}")
                        return result
            
            # Intelligent fallback for action phases - let LLM choose from valid targets
            if game_info['valid_targets']:
                logger.warning(f"No valid target found in response '{response}', using LLM-based selection from {game_info['valid_targets']}")
                
                # Look for any numbers mentioned in the response that are valid targets
                mentioned_numbers = re.findall(r'\b(\d+)\b', response)
                for num_str in mentioned_numbers:
                    target = int(num_str)
                    if target in game_info['valid_targets']:
                        result = f"[{target}]"
                        logger.info(f"Using mentioned target from LLM reasoning: {result}")
                        return result
                
                # This retry mechanism is compliant - LLM still does all reasoning
                try:
                    logger.info("Generating LLM retry with clarified prompt...")
                    retry_prompt = f"""You must choose exactly one target from these valid options: {game_info['valid_targets']}

Your previous response was: "{response}"

Based on your strategy and the game situation, which single target number should you choose?
Respond with ONLY the number in brackets: [X]

Valid targets: {game_info['valid_targets']}"""

                    # Retry generation with focused parameters
                    outputs = self.pipe(
                        retry_prompt,
                        max_new_tokens=10,
                        temperature=0.3,
                        top_k=5,
                        top_p=0.5,
                        do_sample=True,
                        num_return_sequences=1,
                        return_full_text=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    if isinstance(outputs, list) and len(outputs) > 0:
                        retry_response = outputs[0]['generated_text'].strip()
                        
                        # Extract target from retry response
                        retry_match = re.search(r'\[(\d+)\]', retry_response)
                        if retry_match:
                            retry_target = int(retry_match.group(1))
                            if retry_target in game_info['valid_targets']:
                                result = f"[{retry_target}]"
                                logger.info(f"LLM retry successful: {result}")
                                return result
                        
                        # Look for any valid number in the retry response
                        retry_numbers = re.findall(r'\b(\d+)\b', retry_response)
                        for num_str in retry_numbers:
                            target = int(num_str)
                            if target in game_info['valid_targets']:
                                result = f"[{target}]"
                                logger.info(f"LLM retry mentioned target: {result}")
                                return result
                
                except Exception as e:
                    logger.warning(f"LLM retry generation failed: {e}")
                
                # If LLM retry failed, raise exception instead of fallback
                logger.error("LLM failed to generate valid target after retry")
                raise Exception("LLM could not produce valid action format")
            else:
                logger.error("No valid targets available!")
                raise Exception("No valid targets available for action")
        
        # Handle NIGHT_WAIT phase for Villagers with no actions
        elif game_info['phase'] == 'NIGHT_WAIT':
            logger.info("Villager waiting during night phase")
            return "Waiting for day."
        
        # For discussion phase OR when phase is None (assume discussion), return the response as-is (cleaned)
        if game_info['phase'] in ['DAY_DISCUSSION', 'NIGHT_WAIT'] or game_info['phase'] is None:
            # Check if response is a pure action format during discussion (incorrect model output)
            pure_action_match = re.match(r'^\s*\[(\d+)\]\s*$', response.strip())
            if pure_action_match:
                logger.error(f"RESPONSE FORMAT ERROR: Model generated action format '[{pure_action_match.group(1)}]' during DAY_DISCUSSION phase!")
                logger.error(f"This should be natural language discussion, not action format.")
                # Raise exception instead of using fallback response
                raise Exception("LLM generated incorrect format for discussion phase")
            # Clean up the response - remove formatting artifacts
            cleaned = response
            
            # Remove markdown-style formatting
            cleaned = re.sub(r'\*\*[^*]+\*\*:?\s*', '', cleaned)  # Remove **Discussion:** etc.
            # Handle pattern: [Player X] [actual content] - extract the content, not player ID
            player_content_match = re.search(r'\[Player \d+\]\s*\[([^\]]+)\]', cleaned)
            if player_content_match:
                cleaned = player_content_match.group(1)  # Extract actual content
            else:
                # Try to extract any bracket content that's not just a player ID or number
                bracket_matches = re.findall(r'\[([^\]]+)\]', cleaned)
                text_brackets = [m for m in bracket_matches if not re.match(r'^\d+$', m) and not re.match(r'^Player \d+$', m)]
                if text_brackets:
                    # Use the longest text match (likely the actual content)
                    cleaned = max(text_brackets, key=len)
                else:
                    cleaned = re.sub(r'\[.*?\]', '', cleaned)  # Remove all brackets
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            cleaned = cleaned.strip()
            
            if len(cleaned) < 5:  # Too short - raise exception instead of fallback
                raise Exception("LLM generated response too short for discussion phase")
            elif len(cleaned.split()) > 120:  # Too long - increased to 120 words for natural conversation
                words = cleaned.split()
                cleaned = ' '.join(words[:115]) + "..."
            
            logger.info(f"Discussion response: {cleaned}")
            return cleaned
        
        # No more fallbacks - raise exception if we reach here
        logger.error("Reached end of _extract_move without valid response")
        raise Exception("Unable to extract valid move from LLM response")
    
    @modal.method()
    def generate(self, observation: str, tom_insights: str = "", strategic_context: str = "", system_prompt: str = "", phase: str = "discussion") -> Dict:
        """Generate enhanced response with reasoning"""
        request_start = time.time()
        logger.info("=== ENHANCED GENERATION REQUEST ===")
        
        try:
            # Initialize game_info for both paths
            game_info = self._extract_game_info(observation)
            logger.info(f"Game info: {game_info}")
            
            # If system_prompt is provided, use it directly (from streamlined agent)
            if system_prompt:
                logger.info(f"Using provided system prompt for phase: {phase}")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": observation}
                ]
                
                # Set generation parameters based on phase
                if phase == "action":
                    max_tokens = 10
                    temperature = 0.1
                    top_k = 5
                    top_p = 0.3
                else:
                    max_tokens = 80
                    temperature = 0.3
                    top_k = 10
                    top_p = 0.6
                    
            else:
                # Legacy mode - use old prompt system (game_info already extracted above)
                
                # Debug logging for phase detection
                if game_info['phase'] == 'DAY_DISCUSSION':
                    logger.info("DISCUSSION PHASE DETECTED - Expecting natural language response, NOT [number] format")
                elif game_info['phase'] in ['DAY_VOTING', 'NIGHT_MAFIA', 'NIGHT_DOCTOR', 'NIGHT_DETECTIVE']:
                    logger.info(f"ACTION PHASE DETECTED ({game_info['phase']}) - Expecting [number] format only")
                
                # Generate enhanced prompt
                enhanced_prompt = self._generate_enhanced_prompt(observation, game_info, tom_insights, strategic_context)
                
                # Apply chat template
                messages = [
                    {"role": "system", "content": "You are an expert social deduction player."},
                    {"role": "user", "content": enhanced_prompt}
                ]
                
                # Generate with appropriate parameters
                if game_info['phase'] in ['DAY_VOTING', 'NIGHT_MAFIA', 'NIGHT_DOCTOR', 'NIGHT_DETECTIVE']:
                    max_tokens = 10
                    temperature = 0.1
                    top_k = 5
                    top_p = 0.3
                elif game_info['phase'] == 'NIGHT_WAIT':
                    max_tokens = 15
                    temperature = 0.1
                    top_k = 3
                    top_p = 0.2
                else:
                    max_tokens = 80
                    temperature = 0.3
                    top_k = 10
                    top_p = 0.6
            
            try:
                # Use thinking mode for Qwen3-8B enhanced reasoning in complex scenarios
                enable_thinking = phase != "action"  # Enable thinking for discussion, disable for actions
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
                )
            except:
                # Fallback to simple concatenation if chat template fails
                if len(messages) == 2:
                    formatted_prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}"
                else:
                    formatted_prompt = observation
            
            outputs = self.pipe(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            if isinstance(outputs, list) and len(outputs) > 0:
                raw_response = outputs[0]['generated_text'].strip()
            else:
                raw_response = ""
            
            # Handle Qwen direct output format
            raw_response = self._parse_qwen_output(raw_response)
            
            logger.info(f"Raw model output: '{raw_response}'")
            
            # Debug: Check if model generated wrong format for phase
            if game_info['phase'] == 'DAY_DISCUSSION' and re.match(r'^\s*\[\d+\]\s*$', raw_response):
                logger.warning(f"MODEL ERROR: Generated action format '{raw_response}' during DISCUSSION phase!")
            elif game_info['phase'] in ['DAY_VOTING', 'NIGHT_MAFIA', 'NIGHT_DOCTOR', 'NIGHT_DETECTIVE'] and not re.search(r'\[\d+\]', raw_response):
                logger.warning(f"MODEL ERROR: Generated discussion text '{raw_response[:50]}...' during ACTION phase!")
            
            # Extract and format response
            formatted_response = self._extract_move(raw_response, game_info)
            
            total_time = time.time() - request_start
            logger.info(f"Enhanced generation complete in {total_time:.3f}s")
            
            return {
                'response': formatted_response,
                'reasoning': f"Phase: {game_info['phase']}, Role: {game_info['role']}",
                'processing_time': total_time
            }
            
        except Exception as e:
            error_time = time.time() - request_start
            logger.error(f"Enhanced generation error: {e}")
            return {
                'response': "[0]",
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': error_time
            }

# FastAPI app
@app.function(image=image, timeout=300, memory=1024)
@modal.asgi_app()
def fastapi_app():
    web_app = FastAPI(
        title="Enhanced SecretMafia API",
        description="Enhanced social deduction agent for SecretMafia",
        version="2.0.0"
    )
    
    @web_app.post("/generate", response_model=GenerateResponse)
    async def generate_response(request: GenerateRequest):
        """Generate enhanced social deduction response"""
        logger.info("=== ENHANCED API REQUEST ===")
        
        try:
            model = EnhancedSecretMafiaModel()
            result = model.generate.remote(
                request.observation, 
                request.tom_insights, 
                request.strategic_context,
                request.system_prompt,
                request.phase
            )
            
            return GenerateResponse(
                response=result['response'],
                error=result.get('error'),
                error_type=result.get('error_type'),
                processing_time=result.get('processing_time'),
                reasoning=result.get('reasoning')
            )
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return GenerateResponse(
                response="[0]",
                error=str(e),
                error_type=type(e).__name__
            )
    
    @web_app.get("/health")
    async def health_check():
        """Enhanced health check"""
        model_type = "Fine-tuned Qwen3-8B" if USE_FINETUNED_MODEL else "Base Qwen3-8B"
        
        return {
            "status": "healthy",
            "model": f"Enhanced SecretMafia Agent ({model_type})",
            "model_type": model_type,
            "use_finetuned": USE_FINETUNED_MODEL,
            "features": [
                "Strategic role-based reasoning",
                "Phase-aware response generation", 
                "Enhanced prompt engineering",
                "Behavioral pattern analysis",
                "DPO preference learning" if USE_FINETUNED_MODEL else "Inference-time optimization",
                "Self-consistency training" if USE_FINETUNED_MODEL else "Theory of Mind reasoning"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    @web_app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Enhanced SecretMafia Social Deduction API",
            "version": "2.0.0",
            "capabilities": [
                "Advanced social deduction strategies",
                "Role-specific tactical reasoning",
                "Phase-aware communication",
                "Strategic timing and analysis"
            ]
        }
    
    return web_app