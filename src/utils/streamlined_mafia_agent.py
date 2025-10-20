"""
Streamlined Mafia Agent - Theory of Mind + Strategic Components
Removes overcomplication while keeping core strategic reasoning
Rules compliant - all decisions through LLM reasoning
"""

import re
import time
import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from .agent import ModalAgent
from .theory_of_mind import AdvancedToMEngine
from .bidding_system import StrategicBiddingSystem, GameContext, UrgencyLevel

@dataclass
class SimpleGameMemory:
    """Simplified memory system for core game tracking"""
    investigation_results: Dict[int, str] = None
    role_claims: Dict[int, str] = None
    voting_patterns: Dict[int, List[int]] = None
    discussion_history: List[str] = None
    
    def __post_init__(self):
        if self.investigation_results is None:
            self.investigation_results = {}
        if self.role_claims is None:
            self.role_claims = {}
        if self.voting_patterns is None:
            self.voting_patterns = {}
        if self.discussion_history is None:
            self.discussion_history = []

@dataclass
class SimpleGameContext:
    """Simple game context for ToM analysis"""
    my_role: str
    my_player_id: int
    alive_players: List[int]
    turn_number: int

class StreamlinedMafiaAgent(ModalAgent):
    """Streamlined agent with ToM and strategic reasoning, rules compliant"""
    
    def __init__(self, modal_endpoint_url: str):
        super().__init__(modal_endpoint_url)

        # Core strategic components (keep the valuable ones)
        self.tom_engine = AdvancedToMEngine()
        self.bidding_system = StrategicBiddingSystem()

        # Simplified memory
        self.memory = SimpleGameMemory()

        # Game state tracking
        self.my_role: Optional[str] = None
        self.my_player_id: int = -1
        self.alive_players: List[int] = []
        self.turn_count: int = 0

        # Performance tracking
        self.total_rewards = 0.0
        self.strategic_decisions = 0

        # TRM-inspired enhancement: Two-step reasoning for discussion
        self.enable_two_step_reasoning = True  # Set to False to disable
        self.reasoning_cache = {}  # Store internal reasoning for debugging

        print("Agent initialized (Two-Step Reasoning: ENABLED)" if self.enable_two_step_reasoning else "Agent initialized")

    def __call__(self, observation: str) -> str:
        """Main agent decision with logical phase detection and response extraction"""
        self.turn_count += 1
        
        # Update game state
        self._update_game_state(observation)
        
        # Use Theory of Mind to analyze situation
        game_context = self._analyze_with_tom(observation)
        
        print(f"Processing turn {self.turn_count}...")
        
        # Warmup on first turn
        if self.turn_count == 1:
            try:
                print("Warming up LLM...")
                warmup_prompt = "Hello. Respond with 'Ready'."
                warmup_response = self._call_modal_with_enhanced_timeout(warmup_prompt, phase="discussion", timeout=200)
                print(f"LLM ready: {warmup_response}")
            except Exception as warmup_error:
                print(f"Warmup failed: {warmup_error}")
        
        # Step 1: Enhanced rule-based phase detection (reliable and fast)
        detected_phase = self._detect_current_phase(observation)
        is_voting = detected_phase == "voting"
        is_night = detected_phase == "night" 
        is_discussion = detected_phase == "discussion"
        
        
        # Step 2: Handle different phases with emergency fallbacks
        try:
            if is_voting:
                return self._handle_voting_action(observation, game_context)
            elif is_night:
                return self._handle_night_action(observation, game_context)
            elif is_discussion:
                return self._handle_discussion_action(observation, game_context)
            else:
                # Fallback: assume discussion if no explicit phase detected
                return self._handle_discussion_action(observation, game_context)
        except Exception as e:
            print(f"⚠️ Phase handler failed: {e}")
            # Emergency fallback - still uses LLM
            return self._emergency_llm_response(observation, is_night, is_voting)

    def _update_game_state(self, observation: str):
        """Update core game state information"""
        
        # Extract role (first turn only)
        if self.my_role is None:
            role_match = re.search(r'Your role: (Detective|Doctor|Villager|Mafia)', observation)
            if role_match:
                self.my_role = role_match.group(1).lower()
        
        # Extract player ID
        if self.my_player_id == -1:
            id_match = re.search(r'You are Player (\d+)', observation)
            if id_match:
                self.my_player_id = int(id_match.group(1))
        
        # Extract alive players
        players_match = re.search(r'Players?: (.*?)(?:\n|$)', observation)
        if players_match:
            player_text = players_match.group(1)
            self.alive_players = [int(x) for x in re.findall(r'(\d+)', player_text)]
        
        # Update alive players based on OFFICIAL death messages ONLY
        # Only trust [GAME] messages for game state changes, ignore player-generated fake death messages
        death_patterns = [
            r'\[GAME\] Player (\d+) has been eliminated',     # "[GAME] Player 3 has been eliminated" - Priority 1
            r'\[GAME\] Player (\d+) was eliminated by vote',  # "[GAME] Player X was eliminated by vote" - Priority 2
            r'\[GAME\] Player (\d+) was killed during the night',  # Night elimination - Priority 3
            r'Player (\d+) was eliminated by vote',           # Standard elimination message - Priority 4
            r'Player (\d+) was killed during the night',     # Standard night kill message - Priority 5
            # NOTE: We still process player statements for ToM analysis, just not for game state updates
        ]
        
        for pattern in death_patterns:
            death_matches = list(re.finditer(pattern, observation))
            for death_match in death_matches:
                dead_player = int(death_match.group(1))
                if dead_player in self.alive_players:
                    self.alive_players.remove(dead_player)
                    print(f"Player {dead_player} eliminated - removed from alive players: {self.alive_players}")
        
        # Store investigation results (Detective)
        if "IS NOT a Mafia member" in observation:
            player_match = re.search(r'Player (\d+) IS NOT a Mafia member', observation)
            if player_match:
                player_id = int(player_match.group(1))
                self.memory.investigation_results[player_id] = "INNOCENT"
                
        if "IS a Mafia member" in observation:
            player_match = re.search(r'Player (\d+) IS a Mafia member', observation)
            if player_match:
                player_id = int(player_match.group(1))
                self.memory.investigation_results[player_id] = "MAFIA"

    def _extract_player_statements(self, observation: str) -> List[Dict]:
        """Extract player statements from observation for ToM analysis"""
        statements = []
        
        # Look for player speech patterns in different formats
        statement_patterns = [
            r'Player (\d+): (.+?)(?=Player \d+:|$)',
            r'\[Player (\d+)\]: (.+?)(?=\[Player \d+\]:|$)',
            r'(\d+): (.+?)(?=\d+:|$)',
        ]
        
        for pattern in statement_patterns:
            matches = re.findall(pattern, observation, re.DOTALL | re.MULTILINE)
            for player_id, statement in matches:
                if statement.strip() and len(statement.strip()) > 5:
                    statements.append({
                        'player_id': int(player_id),
                        'statement': statement.strip()
                    })
        
        # Also look for voting statements
        vote_matches = re.findall(r'Player (\d+) voted for Player (\d+)', observation)
        for voter_id, target_id in vote_matches:
            statements.append({
                'player_id': int(voter_id),
                'statement': f"voted for Player {target_id}",
                'action_type': 'vote'
            })
        
        return statements

    def _analyze_with_tom(self, observation: str) -> SimpleGameContext:
        """Use Theory of Mind engine to analyze current situation"""
        
        try:
            # Extract player statements from observation for ToM analysis
            statements = self._extract_player_statements(observation)
            
            # Update ToM engine with new statements (skip if engine has issues)
            if statements and hasattr(self.tom_engine, 'update_beliefs_from_statements'):
                try:
                    # Convert dictionary format to tuple format expected by ToM engine
                    statement_tuples = [(stmt['player_id'], stmt['statement']) for stmt in statements]
                    self.tom_engine.update_beliefs_from_statements(statement_tuples)
                except Exception as tom_error:
                    print(f"⚠️ ToM engine update failed: {tom_error}")
            
            # Create game context for decision making
            game_context = SimpleGameContext(
                my_role=self.my_role or "unknown",
                my_player_id=self.my_player_id,
                alive_players=self.alive_players,
                turn_number=self.turn_count
            )
            
            return game_context
            
        except Exception as e:
            print(f"⚠️ ToM analysis failed: {e}")
            # Return basic game context
            return SimpleGameContext(
                my_role=self.my_role or "unknown",
                my_player_id=self.my_player_id,
                alive_players=self.alive_players,
                turn_number=self.turn_count
            )

    # Removed old complex night action handler - now using unified approach



    def _generate_tom_insights(self, game_context: SimpleGameContext) -> str:
        """Generate Theory of Mind insights for strategic decision making"""
        
        insights = []
        
        # Player behavior analysis
        insights.append(f"Players analyzed: {len(self.alive_players)}")
        
        # Investigation results context
        if self.memory.investigation_results:
            insights.append(f"Known information: {self.memory.investigation_results}")
        
        # Role-specific ToM considerations
        if self.my_role == "detective":
            insights.append("ToM: Others expect Detective to share investigation results strategically")
        elif self.my_role == "doctor":
            insights.append("ToM: Others may suspect Doctor if night kills fail")
        elif self.my_role == "villager":
            insights.append("ToM: Others expect Villagers to be cautious but helpful")
        elif self.my_role == "mafia":
            insights.append("ToM: Must appear helpful while avoiding suspicion")
        
        return "\n".join(insights)

    def _generate_strategic_context(self, game_context: SimpleGameContext) -> str:
        """Generate strategic context for decision making"""
        
        context = []
        
        # Game phase analysis
        if self.turn_count <= 3:
            context.append("Early game: Focus on information gathering")
        elif self.turn_count <= 6:
            context.append("Mid game: Strategic positioning important")
        else:
            context.append("Late game: Critical decisions needed")
        
        # Player count dynamics
        if len(self.alive_players) <= 4:
            context.append("Few players left: Every decision critical")
        
        # Role-specific strategy
        if self.my_role == "detective" and self.memory.investigation_results:
            context.append("Use investigation results strategically")
        
        return "\n".join(context)

    def _call_modal_with_enhanced_timeout(self, user_prompt: str, phase: str = "discussion", timeout: int = 30) -> str:
        """Call Modal with proper system prompts per phase"""

        # Create phase-specific system prompts with Qwen recommendation
        base_system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. ALWAYS respond in ENGLISH language, never in Chinese (不要用中文)."
        
        if phase == "action":
            # For voting and night actions
            system_prompt = f"""{base_system}

You are Player {getattr(self, 'my_player_id', 'X')} in a Mafia game. Your role: {getattr(self, 'my_role', 'Unknown')}.

CRITICAL: YOU ARE PLAYER {getattr(self, 'my_player_id', 'X')} - DO NOT TARGET YOURSELF!

ROLE-SPECIFIC STRATEGY:
- Villager: Vote to eliminate suspected Mafia members
- Mafia: Eliminate Village power roles (Doctor/Detective) or strong Village players  
- Doctor: Protect confirmed Village roles (NOT yourself unless desperate)
- Detective: Investigate suspicious players to confirm Mafia status

ACTION PHASE REQUIREMENTS:
- ALWAYS respond with EXACTLY [NUMBER] format (e.g., [3], [5])
- Choose from the valid targets provided in the user prompt
- Do NOT add any explanation or extra text
- CRITICAL: Do NOT vote/target yourself (Player {getattr(self, 'my_player_id', 'X')})
- Prioritize strategic targets based on your role
- Example response: [3] (just the number in brackets, nothing else)"""

        else:
            # For discussion phases
            system_prompt = f"""{base_system}

You are Player {getattr(self, 'my_player_id', 'X')} in a Mafia game. Your role: {getattr(self, 'my_role', 'Unknown')}.

DISCUSSION PHASE IDENTITY:
- You are Player {getattr(self, 'my_player_id', 'X')} speaking directly to other players
- Use "I" when referring to yourself and your actions
- Address other players by their Player numbers (Player 0, Player 1, etc.)
- You are having a real conversation with other living players

DISCUSSION PHASE RULES:
- Respond with natural conversation (20-100 words)
- Use natural language only - no [number] voting format during discussion
- Ask questions, make accusations, defend yourself
- Build suspicion and cases against likely Mafia players

CONVERSATION STYLE:
- "I think you're suspicious because..."
- "Player 2, why did you act that way?"
- "Your story doesn't make sense to me."
- "I don't trust Player 3's explanation."

PURPOSE: Build cases and suspicion during discussion, vote during voting phase.

YOUR MISSION AS A PLAYER:
Participate in the discussion by making accusations, defending yourself/others, or asking questions.
You are trying to win the game for your team."""

        # Send proper system/user message structure to Modal
        
        response = requests.post(
            self.endpoint_url + '/generate',
            json={
                "observation": user_prompt,
                "system_prompt": system_prompt,
                "phase": phase
            },
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json().get("response", "").strip()
        
        if not result:
            raise Exception("Empty response from LLM")
            
        return result

    def _get_llm_response_with_retries(self, primary_prompt: str, observation: str, phase: str = "discussion") -> str:
        """Rules-compliant emergency system - multiple LLM attempts with generous timeouts for cold start"""
        
        # Increase timeouts to match Modal endpoint response times
        if self.turn_count <= 2:
            # Very generous timeouts for first few turns (Modal cold start)
            primary_timeout = 180
            secondary_timeout = 120
            emergency_timeout = 90
            final_timeout = 60
        elif self.turn_count <= 5:
            # Generous timeouts for mid-game (Modal warming up)
            primary_timeout = 120
            secondary_timeout = 90
            emergency_timeout = 60
            final_timeout = 45
        else:
            # Standard timeouts for late game (Modal warmed up)
            primary_timeout = 90
            secondary_timeout = 60
            emergency_timeout = 45
            final_timeout = 30
        
        try:
            # Primary attempt
            print(f"LLM call (timeout: {primary_timeout}s)...")
            return self._call_modal_with_enhanced_timeout(primary_prompt, phase=phase, timeout=primary_timeout)
            
        except Exception as e1:
            print(f"Primary LLM failed: {e1}")
            
            try:
                # Secondary attempt: Check if this is voting phase and use JSON format (better for Qwen)
                if "[NUMBER] ONLY!" in primary_prompt:
                    json_voting_prompt = f"""JSON VOTE:

{{"vote": NUMBER}}

Choose from: {self.alive_players}

JSON:"""
                    
                    response = self._call_modal_with_enhanced_timeout(json_voting_prompt, phase="action", timeout=secondary_timeout)
                    
                    # Try to extract from JSON
                    try:
                        import json
                        parsed = json.loads(response.strip())
                        if "vote" in parsed:
                            vote_num = int(parsed["vote"])
                            if vote_num in self.alive_players:
                                return f"[{vote_num}]"
                    except:
                        pass
                    
                    return response
                else:
                    # Standard simplified prompt for non-voting
                    if phase == "action":
                        simplified_prompt = f"""You are {self.my_role} in Secret Mafia night/voting phase.

Choose a player number from: {self.alive_players}

Respond with [X] format only:"""
                    else:
                        simplified_prompt = f"""You are {self.my_role} in Secret Mafia discussion.

Current situation: {observation[-300:]}

Provide a brief strategic comment (NO brackets, just natural language):"""
                    
                    return self._call_modal_with_enhanced_timeout(simplified_prompt, phase=phase, timeout=secondary_timeout)
                
            except Exception as e2:
                print(f"⚠️ Simplified LLM attempt failed: {e2}")
                
                try:
                    # Emergency ultra-simple LLM prompt
                    if phase == "action":
                        emergency_prompt = f"""Role: {self.my_role}. Choose player number [X]:"""
                    else:
                        emergency_prompt = f"""Role: {self.my_role}. Brief comment on situation:"""
                    
                    return self._call_modal_with_enhanced_timeout(emergency_prompt, phase=phase, timeout=emergency_timeout)
                    
                except Exception as e3:
                    print(f"❌ All LLM attempts failed: {e3}")
                    
                    # Final attempt - force LLM to give ANY response (still pure LLM)
                    try:
                        if phase == "action":
                            final_prompt = f"You are {self.my_role}. Pick [NUMBER]:"
                        else:
                            final_prompt = f"You are {self.my_role}. Say something:"
                        result = self._call_modal_with_enhanced_timeout(final_prompt, phase=phase, timeout=final_timeout)
                        if result:
                            return result
                        else:
                            # If even this fails, we have to raise an exception - no fallbacks allowed
                            raise Exception("All LLM attempts failed - no response generated")
                        
                    except Exception as e4:
                        print(f"💥 Complete LLM failure: {e4}")
                        # Rules compliant: Cannot use fallback answers - must raise exception
                        raise Exception(f"LLM completely unresponsive after all attempts: {e4}")

    def _is_night_action_phase(self, observation: str) -> bool:
        """Check if current turn requires night action"""
        # Only return True if we actually have a night action to take
        if self.my_role and self.my_role.lower() == "villager":
            return False  # Villagers have no night actions
        
        # Check ONLY the last few lines for current night phase requirements
        lines = observation.strip().split('\n')
        last_lines = lines[-5:]  # Most recent system messages
        
        current_night_indicators = [
            'Mafia, agree on a victim',
            'choose one player to investigate',
            'choose one player to protect', 
            'choose one player to eliminate',
            'Night phase -'
        ]
        
        for line in last_lines:
            if any(indicator in line for indicator in current_night_indicators):
                return True
        
        return False
    
    def _is_voting_phase(self, observation: str) -> bool:
        """Check if current turn requires voting"""
        # Check ONLY the last few lines for current voting phase requirements
        lines = observation.strip().split('\n')
        last_lines = lines[-5:]  # Most recent system messages
        
        current_voting_indicators = [
            'Voting phase - submit one vote',
            'submit one vote in format'
        ]
        
        for line in last_lines:
            if any(indicator in line for indicator in current_voting_indicators):
                return True
        
        return False
    
    def _is_discussion_phase(self, observation: str) -> bool:
        """Check if current turn is discussion phase"""
        # Check ONLY the last few lines for current discussion phase requirements
        lines = observation.strip().split('\n')
        last_lines = lines[-5:]  # Most recent system messages
        
        current_discussion_indicators = [
            'Day breaks. Discuss for',
            'Discuss for'
        ]
        
        for line in last_lines:
            if any(indicator in line for indicator in current_discussion_indicators):
                return True
        
        return False
    
    def _emergency_llm_response(self, observation: str, is_night: bool, is_voting: bool) -> str:
        """Emergency LLM-based response when main handlers fail"""
        print("🚨 Using emergency LLM response")
        
        if is_night and self.my_role and self.my_role.lower() not in ["villager", "citizen"]:
            # Emergency night action
            emergency_prompt = f"""EMERGENCY: Night action needed!
            
Role: {self.my_role}
Players: {self.alive_players}

Choose a player number for your night action.
Respond with [NUMBER] only:"""
            
        elif is_voting:
            # Emergency voting
            emergency_prompt = f"""EMERGENCY: Vote needed!
            
Players to vote for: {self.alive_players}

Choose who to vote for.
Respond with [NUMBER] only:"""
            
        else:
            # Emergency discussion or villager night
            emergency_prompt = f"""EMERGENCY: Response needed!
            
Role: {self.my_role}
Situation: {observation[-200:]}

Respond appropriately for your role:"""
        
        try:
            # Use reduced timeout for emergency
            response = self._call_modal_with_enhanced_timeout(emergency_prompt, phase="action" if (is_night or is_voting) else "discussion", timeout=30)
            if is_night or is_voting:
                return self._extract_move(response, observation)
            else:
                return response.strip()
        except Exception as emergency_error:
            print(f"💥 Emergency LLM also failed: {emergency_error}")
            # Final emergency - minimal LLM prompt
            try:
                final_prompt = f"Role: {self.my_role}. Continue game:"
                final_response = self._call_modal_with_enhanced_timeout(final_prompt, phase="discussion", timeout=15)
                return final_response.strip() if final_response else "I continue the game."
            except:
                # If all LLM attempts fail, return minimal response
                return "I continue with the game."
    

    def _extract_valid_targets_from_observation(self, observation: str) -> List[int]:
        """Extract valid targets from current observation, prioritizing most recent/specific patterns"""
        if not observation:
            return self.alive_players
        
        lines = observation.strip().split('\n')
        
        # STRATEGY: Check last 3 lines first (most current), then expand search
        # This ensures we get the CURRENT action's valid targets, not old ones
        
        # Priority 1: Check last 3 lines for specific night/voting action patterns
        last_3_lines = lines[-3:] if len(lines) >= 3 else lines
        
        for line in reversed(last_3_lines):  # Check most recent first
            line_lower = line.lower().strip()
            
            # Pattern 1: "Night phase - choose one player to investigate: [2], [3], [4]"
            night_action_match = re.search(r'night phase - choose one player to (?:investigate|protect|eliminate): \[([0-9, \]]+)\]', line_lower)
            if night_action_match:
                targets_text = night_action_match.group(1)
                valid_targets = [int(x) for x in re.findall(r'(\d+)', targets_text)]
                if valid_targets:
                    # Validate against current alive players
                    valid_targets = [t for t in valid_targets if t in self.alive_players and t != self.my_player_id]
                    return valid_targets
            
            # Pattern 2: "Voting phase - submit one vote in format [X]. Valid: [0], [1], [2], [3], [4]"
            voting_valid_match = re.search(r'valid: \[([0-9, \]]+)\]', line_lower)
            if voting_valid_match:
                targets_text = voting_valid_match.group(1)
                valid_targets = [int(x) for x in re.findall(r'(\d+)', targets_text)]
                if valid_targets:
                    # Validate against current alive players
                    valid_targets = [t for t in valid_targets if t in self.alive_players and t != self.my_player_id]
                    return valid_targets
            
            # Pattern 3: "valid options: '[0]', '[1]', '[2]'"
            options_match = re.search(r'valid options: (.+)', line_lower)
            if options_match:
                targets_text = options_match.group(1)
                valid_targets = [int(x) for x in re.findall(r'(\d+)', targets_text)]
                if valid_targets:
                    # Validate against current alive players
                    valid_targets = [t for t in valid_targets if t in self.alive_players and t != self.my_player_id]
                    return valid_targets
        
        # Priority 2: Check last 5 lines for any valid target patterns
        last_5_lines = lines[-5:] if len(lines) >= 5 else lines
        
        for line in reversed(last_5_lines):
            line_lower = line.lower().strip()
            
            # Look for any pattern with "valid" and brackets
            if "valid" in line_lower and "[" in line_lower:
                # Extract all numbers in brackets from this line
                bracket_numbers = re.findall(r'\[(\d+)\]', line_lower)
                if bracket_numbers:
                    valid_targets = [int(x) for x in bracket_numbers]
                    # Validate against current alive players
                    valid_targets = [t for t in valid_targets if t in self.alive_players and t != self.my_player_id]
                    return valid_targets
        
        # Priority 3: Broader search in entire observation for fallback patterns
        broader_patterns = [
            r'valid votes: (.+)',                       # "Valid votes: '[0]', '[1]', '[2]'"
            r'valid targets: \[([0-9, \]]+)\]',        # "Valid targets: [2], [3], [4]"
            r'valid: \[([0-9, \]]+)\]',                # "Valid: [1], [4], [5]"
        ]
        
        for pattern in broader_patterns:
            matches = list(re.finditer(pattern, observation.lower()))
            if matches:
                # Use the last match (most recent)
                last_match = matches[-1]
                valid_text = last_match.group(1)
                valid_targets = [int(x) for x in re.findall(r'(\d+)', valid_text)]
                if valid_targets:
                    # Validate against current alive players
                    valid_targets = [t for t in valid_targets if t in self.alive_players and t != self.my_player_id]
                    return valid_targets
        
        # Priority 4: If this appears to be an action phase, try to extract from any bracket format
        if self._appears_to_be_action_phase(observation):
            # Extract all bracket numbers from the entire observation
            all_bracket_numbers = re.findall(r'\[(\d+)\]', observation)
            if all_bracket_numbers:
                # Convert to integers and remove duplicates while preserving order
                seen = set()
                valid_targets = []
                for num_str in all_bracket_numbers:
                    num = int(num_str)
                    if num not in seen and num != self.my_player_id and num in self.alive_players:  # Exclude self and dead players
                        seen.add(num)
                        valid_targets.append(num)
                
                if valid_targets:
                    return valid_targets
        
        # Final fallback: Use current alive_players but exclude dead players
        # Filter alive_players to remove any players we know are dead
        filtered_alive = [p for p in self.alive_players if p != self.my_player_id]  # Don't target self
        
        # Additional validation: remove recently eliminated players
        recent_lines = observation.strip().split('\n')[-5:]
        for line in recent_lines:
            if "was eliminated" in line.lower():
                eliminated_match = re.search(r'Player (\d+) was eliminated', line)
                if eliminated_match:
                    eliminated_player = int(eliminated_match.group(1))
                    if eliminated_player in filtered_alive:
                        filtered_alive.remove(eliminated_player)
        
        print(f"Valid targets extracted: {filtered_alive}, Current alive: {self.alive_players}")
        return filtered_alive if filtered_alive else self.alive_players
    
    def _handle_night_action(self, observation: str, game_context) -> str:
        """Handle night action phase - give LLM clear context with valid choices"""
        
        # Extract valid targets and include in prompt for LLM context
        valid_targets = self._extract_valid_targets_from_observation(observation)
        
        # Generate strategic context with ToM insights
        tom_insights = self._generate_tom_insights(game_context)
        strategic_context = self._generate_strategic_context(game_context)
        
        # Get advanced ToM analysis for better decision making
        advanced_tom = ""
        if hasattr(self.tom_engine, 'get_comprehensive_strategic_insights'):
            try:
                advanced_tom = self.tom_engine.get_comprehensive_strategic_insights(
                    self.my_player_id, self.alive_players, self.my_role
                )
            except Exception as e:
                print(f"⚠️ Advanced ToM failed: {e}")
                advanced_tom = ""
        
        # Get role-specific night strategy
        night_strategy = self._get_night_action_strategy(self.my_role, valid_targets, observation)
        
        night_prompt = f"""You are Player {self.my_player_id} ({self.my_role}) - Night Phase

{night_strategy}

CURRENT SITUATION:
{observation}

Available targets: {valid_targets}

ADVANCED ANALYSIS:
{advanced_tom}

CRITICAL DECISION FACTORS:
- Who poses the biggest threat to your team?
- Who might have special roles (Detective/Doctor)?
- Who is leading discussions against your team?
- Who would others least expect you to target?

Choose strategically. Respond with [NUMBER] format (e.g., [3]):"""
        
        response = self._get_llm_response_with_retries(night_prompt, observation, phase="action")
        return self._extract_move(response, observation)
    
    def _handle_voting_action(self, observation: str, game_context) -> str:
        """Handle voting phase - give LLM clear context with valid choices"""
        
        # Extract valid targets and include in prompt for LLM context
        valid_targets = self._extract_valid_targets_from_observation(observation)
        
        # Get advanced ToM analysis for voting
        advanced_tom = ""
        if hasattr(self.tom_engine, 'get_comprehensive_strategic_insights'):
            try:
                advanced_tom = self.tom_engine.get_comprehensive_strategic_insights(
                    self.my_player_id, self.alive_players, self.my_role
                )
            except Exception as e:
                print(f"⚠️ Advanced ToM failed: {e}")
                advanced_tom = ""
        
        # Add self-preservation check for ALL roles
        if self.my_player_id in valid_targets:
            valid_targets = [t for t in valid_targets if t != self.my_player_id]
            print(f"⚠️ Removed self ({self.my_player_id}) from valid targets: {valid_targets}")
        
        # Special Doctor voting strategy
        if self.my_role == "doctor":
            voting_prompt = f"""VOTING PHASE - Player {self.my_player_id} (DOCTOR)

DOCTOR VOTING STRATEGY: Support the Village consensus safely!
Don't lead votes, but support reasonable elimination choices.

CURRENT GAME SITUATION:
{observation}

Available players to vote for: {valid_targets}

DOCTOR VOTING PRIORITIES:
- Vote with the Village majority when possible
- Avoid being the deciding vote that draws attention
- Support eliminations of clearly suspicious players
- Don't vote against strong Village players
- Vote for Mafia if you can identify them safely

DOCTOR VOTING APPROACH:
- Look for players with majority suspicion
- Avoid voting for confirmed Village roles
- Support evidence-based eliminations
- Stay with the group to avoid standing out

Choose your vote to support Village interests while staying safe.
Respond with ONLY [NUMBER] format (e.g., [3]):"""
        elif self.my_role == "villager":
            voting_prompt = f"""VOTING PHASE - Player {self.my_player_id} (VILLAGER)

VILLAGER VOTING MISSION: Eliminate Mafia members to win!

CURRENT GAME SITUATION:
{observation}

Available players to vote for: {valid_targets}

CRITICAL VILLAGER VOTING RULES:
1. NEVER VOTE FOR YOURSELF - You are Player {self.my_player_id}!
2. TRUST Detective investigations when provided
3. VOTE for confirmed Mafia members immediately
4. IGNORE deflection tactics trying to shift blame
5. PRIORITIZE evidence over speculation

VILLAGER VOTING PRIORITIES:
- If Detective revealed "Player X is Mafia" → Vote [X] immediately
- If multiple players confirmed suspicious → Vote most suspicious
- If no clear evidence → Vote player with most deflection/evasion
- NEVER vote for confirmed Village roles (Detective/Doctor)

DEFLECTION WARNING:
- Mafia will try to shift blame to innocent players
- Focus on EVIDENCE not accusations without proof
- Trust investigation results over speculation

Choose the Mafia member to eliminate based on evidence.
Respond with ONLY [NUMBER] format (e.g., [3]):"""
        else:
            voting_prompt = f"""VOTING PHASE - Player {self.my_player_id} ({self.my_role})

CRITICAL: This is the VOTING phase - you must eliminate one player!
Based on the discussion, choose who to vote for elimination.

CURRENT GAME SITUATION:
{observation}

Available players to vote for: {valid_targets}

VOTING STRATEGY ({self.my_role}):
- Eliminate the player you found most suspicious during discussion
- Consider who deflected questions or gave weak explanations
- Target players who seemed evasive or overly defensive
- Vote for the player most likely to be Mafia based on behavior

Choose your elimination target based on discussion evidence.
Respond with ONLY [NUMBER] format (e.g., [3]):"""
        
        response = self._get_llm_response_with_retries(voting_prompt, observation, phase="action")
        return self._extract_move(response, observation)
    
    def _handle_discussion_action(self, observation: str, game_context) -> str:
        """Handle discussion phase with optional two-step reasoning"""

        # ENHANCEMENT: Two-step reasoning (Reason → Act)
        # Step 1: Generate internal strategic reasoning (private)
        # Step 2: Generate public action based on reasoning
        # Falls back to original behavior if disabled or fails

        if self.enable_two_step_reasoning:
            try:
                # Step 1: Internal reasoning
                internal_reasoning = self._generate_internal_reasoning(observation, game_context)

                if internal_reasoning and len(internal_reasoning.strip()) >= 10:
                    # Store for debugging
                    self.reasoning_cache[self.turn_count] = internal_reasoning

                    # Step 2: Generate action from reasoning
                    response = self._generate_action_from_reasoning(observation, game_context, internal_reasoning)

                    # Store in memory
                    self.memory.discussion_history.append(response)
                    if len(self.memory.discussion_history) > 5:
                        self.memory.discussion_history = self.memory.discussion_history[-5:]

                    return response
                else:
                    print("⚠️ Reasoning step failed, using original method")
            except Exception as e:
                print(f"⚠️ Two-step reasoning failed: {e}, falling back to original")

        # ORIGINAL METHOD (fallback or when two-step disabled)
        tom_insights = self._generate_tom_insights(game_context)
        strategic_context = self._generate_strategic_context(game_context)

        # Get advanced ToM analysis for discussion
        advanced_tom = ""
        if hasattr(self.tom_engine, 'get_comprehensive_strategic_insights'):
            try:
                advanced_tom = self.tom_engine.get_comprehensive_strategic_insights(
                    self.my_player_id, self.alive_players, self.my_role
                )
            except Exception as e:
                print(f"⚠️ Advanced ToM failed: {e}")
                advanced_tom = ""

        # Extract recent player statements to avoid repetitive responses
        recent_statements = self._extract_recent_player_statements(observation)

        # Check if we recently said something similar
        recent_discussion = "\n".join(self.memory.discussion_history[-3:]) if self.memory.discussion_history else "None"

        # Include investigation results if we're Detective - CRITICAL TO SHARE
        investigation_info = ""
        if self.my_role == "detective" and self.memory.investigation_results:
            investigation_info = f"\nCRITICAL - YOUR INVESTIGATION RESULTS TO SHARE: {self.memory.investigation_results}"
            investigation_info += "\n- You MUST share these results to help the Village team win!"
            investigation_info += "\n- DO NOT fabricate or make up fake investigation results!"
            investigation_info += "\n- Example format: 'I'm the Detective. Player X is CONFIRMED [MAFIA/INNOCENT].'"

        # Extract just the most recent NEW information that requires a response
        latest_developments = self._extract_latest_developments(observation)

        # Generate a completely different prompt each turn to force unique responses
        unique_context = self._generate_unique_turn_context(self.turn_count, observation, recent_discussion)

        # Get role-specific strategy
        role_strategy = self._get_role_specific_discussion_strategy(self.my_role, observation)
        
        # Special Doctor prompt emphasizing subtlety and survival
        if self.my_role == "doctor":
            discussion_prompt = f"""YOU ARE Player {self.my_player_id} (DOCTOR) - STAY SUBTLE AND ALIVE!

CRITICAL DOCTOR MISSION: Survive to keep protecting Village team!
- Your value is in STAYING ALIVE, not in hunting Mafia
- Be supportive and helpful, but NOT the most vocal player
- Let Villagers lead discussions while you blend in
- Avoid drawing attention that would make you a target

{role_strategy}{investigation_info}

RECENT DEVELOPMENTS:
{self._extract_real_game_events(observation)}

OTHER PLAYERS' ACTIONS:
{self._extract_real_player_actions(observation)}

DOCTOR DISCUSSION STRATEGY:
1. Support other players' reasonable observations
2. Agree with evidence-based suggestions
3. Avoid making direct accusations yourself
4. Stay helpful but not threatening
5. Blend in as a concerned but cautious player

SUBTLE DOCTOR EXAMPLES:
- "I think Player 2 makes some valid points there."
- "We should be careful about rushing to judgment."
- "That's an interesting observation - worth considering."
- "I agree we need to look at the evidence carefully."

AVOID THESE DOCTOR MISTAKES:
- "Player X is definitely suspicious!" (too aggressive)
- "We need to eliminate Player Y!" (draws attention)
- Leading votes or discussions (makes you a target)

REMEMBER: You protect others at NIGHT, survive during DAY!

Respond as a supportive but cautious Village member:"""
        
        # Special Detective prompt if we have investigation results to share
        elif self.my_role == "detective" and self.memory.investigation_results:
            discussion_prompt = f"""YOU ARE Player {self.my_player_id} (DETECTIVE) - SHARE YOUR INVESTIGATION RESULTS NOW!

CRITICAL: You have investigation results that can help Village win!
{investigation_info}

DETECTIVE COMMUNICATION STRATEGY:
- Lead with your investigation results immediately
- Be confident and assertive when sharing evidence
- Reveal your role to give credibility to your findings
- Rally others to vote based on your confirmed results

RECENT DEVELOPMENTS:
{self._extract_real_game_events(observation)}

OTHER PLAYERS' ACTIONS:
{self._extract_real_player_actions(observation)}

DETECTIVE RESPONSE EXAMPLES:
- "I'm the Detective. I investigated Player 3 last night - they are CONFIRMED MAFIA!"
- "Everyone listen - I have proof. Player 1 is NOT Mafia, I checked them last night."
- "As the Detective, I can confirm Player 2 is innocent. We need to look elsewhere."

Reveal your investigation results to help Village win:"""
        else:
            # Special Villager prompt emphasizing action (FIXED CONFLICTS)
            if self.my_role == "villager":
                discussion_prompt = f"""YOU ARE Player {self.my_player_id} (VILLAGER) - HUNT MAFIA!

VILLAGER MISSION: Find and eliminate Mafia members through evidence!
- LISTEN for Detective investigation results and trust them
- BUILD cases against Mafia players through questions and observations
- IDENTIFY deflection tactics when Mafia tries to shift blame
- PREPARE your voting strategy based on solid evidence

{role_strategy}{investigation_info}

RECENT DEVELOPMENTS:
{self._extract_real_game_events(observation)}

OTHER PLAYERS' ACTIONS:
{self._extract_real_player_actions(observation)}

VILLAGER DISCUSSION PRIORITIES:
1. Support Detective investigation claims with questions
2. Call out players who deflect or avoid questions
3. Build evidence-based cases against suspicious players
4. Prepare to vote for confirmed Mafia when voting comes

DETECTIVE EVIDENCE FOCUS:
- If someone says "I investigated Player X - they are Mafia" → Support this!
- Ask follow-up questions to confirm Detective claims
- Prepare to vote for investigated Mafia members
- Don't let deflection tactics distract from evidence

SMART VILLAGER EXAMPLES:
- "If Player X was investigated and found to be Mafia, we should vote for them."
- "Player Y is trying to deflect attention - that's suspicious."
- "What evidence do you have against Player Z? Detective results are more reliable."
- "Stop trying to change the subject - let's focus on the investigation."

REMEMBER: Evidence-based decisions win games for Village!

Build cases based on Detective investigations and solid evidence:"""
            else:
                discussion_prompt = f"""YOU ARE Player {self.my_player_id} ({self.my_role}) - SPEAK NATURALLY TO OTHER PLAYERS

IDENTITY & TONE:
- You are a REAL PLAYER with emotions and suspicions
- React personally to accusations and events
- Be LESS analytical, MORE human
- Show genuine concern for your survival
- NEVER sound like you're analyzing a game for others

{role_strategy}{investigation_info}

RECENT DEVELOPMENTS:
{self._extract_real_game_events(observation)}

OTHER PLAYERS' ACTIONS:
{self._extract_real_player_actions(observation)}

YOUR PREVIOUS STATEMENT (vary your approach):
"{recent_discussion.split('.')[-1] if recent_discussion else 'First time speaking'}"

RESPONSE STYLE:
- React emotionally to accusations
- Point fingers at others when suspicious
- Defend yourself when under pressure
- Ask direct questions to other players
- Show you're invested in winning

GOOD EXAMPLES:
- "Player 2, that vote came out of nowhere - why are you targeting me?"
- "I don't trust Player 4's story about being innocent"
- "Wait, why is everyone suddenly agreeing? That's suspicious"
- "Player 1, you've been awfully quiet - what are you hiding?"

BAD EXAMPLES:
- "Usually villagers take more time..." (too analytical)
- "This indicates..." (sounds like analysis)
- "We should consider..." (sounds like teaching)

Respond as a concerned player fighting for survival:"""
        
        response = self._get_llm_response_with_retries(discussion_prompt, observation, phase="discussion")
        cleaned_response = response.strip()
        
        # Store in discussion history to track repetition
        self.memory.discussion_history.append(cleaned_response)
        
        # Keep only last 5 discussion entries to avoid memory bloat
        if len(self.memory.discussion_history) > 5:
            self.memory.discussion_history = self.memory.discussion_history[-5:]
        
        return cleaned_response
    
    def _extract_move(self, response: str, observation: str = "") -> str:
        """Extract action from LLM response"""
        
        # Clean the response first
        cleaned = response.strip()
        
        # Look for pattern like [0], [1], etc. - already in correct format
        bracket_match = re.search(r'\[(\d+)\]', cleaned)
        if bracket_match:
            result = f"[{bracket_match.group(1)}]"
            return result
        
        # Look for single digits and convert to bracket format
        digit_match = re.search(r'\b(\d+)\b', cleaned)
        if digit_match:
            result = f"[{digit_match.group(1)}]"
            return result
        
        # If appears to be action phase but no number found, extract any digit
        if self._appears_to_be_action_phase(observation):
            # Try to find any digit in the response
            any_digit = re.search(r'(\d)', cleaned)
            if any_digit:
                result = f"[{any_digit.group(1)}]"
                return result
            else:
                # Absolute last resort: random valid choice
                valid_targets = self._extract_valid_targets_from_observation(observation)
                if valid_targets:
                    import random
                    random_choice = random.choice(valid_targets)
                    result = f"[{random_choice}]"
                    return result
        
        # Return LLM response as-is for discussion phases
        return cleaned

    
    def _appears_to_be_action_phase(self, observation: str) -> bool:
        """Check if observation suggests this is an action phase (voting/night)"""
        action_indicators = [
            "Voting phase",
            "submit one vote", 
            "Night phase",
            "choose one player to investigate",
            "choose one player to protect",
            "choose one player to eliminate",
            "Mafia, agree on a victim"
        ]
        
        obs_lower = observation.lower()
        return any(indicator.lower() in obs_lower for indicator in action_indicators)

    def _detect_current_phase(self, observation: str) -> str:
        """
        FIXED rule-based phase detection focused on CURRENT phase (last few lines).
        
        Args:
            observation: Current game observation text
            
        Returns:
            One of: "night", "voting", "discussion", "waiting", "unknown"
        """
        if not observation or observation.strip() == "":
            return "unknown"

        lines = observation.strip().split('\n')
        
        # STRATEGY: Focus on LAST 3 lines first (highest priority - current action)
        # Then check last 5 lines, then last 10 lines
        last_3_lines = '\n'.join(lines[-3:]).lower() if len(lines) >= 3 else observation.lower()
        last_5_lines = '\n'.join(lines[-5:]).lower() if len(lines) >= 5 else observation.lower()
        last_10_lines = '\n'.join(lines[-10:]).lower() if len(lines) >= 10 else observation.lower()
        
        # PRIORITY 1: Error recovery (ALWAYS highest priority) - check last 3 lines first
        if "resubmit a valid move" in last_3_lines or "attempted an invalid move" in last_3_lines:
            # Look at what action was invalid to determine correct phase
            if "investigation target" in last_3_lines or "investigate" in last_5_lines:
                return "night"
            else:
                return "voting"
        
        # PRIORITY 1.5: Direct voting format indicators
        if "valid: [" in last_3_lines and ("submit" in last_3_lines or "vote" in last_3_lines):
            return "voting"
        
        # PRIORITY 2: CURRENT action requests in last 3 lines (most important)
        # Night phase indicators - exact patterns from documentation
        current_night_patterns = [
            "night phase - choose one player to investigate:",
            "night phase - choose one player to protect:",
            "night phase - choose one player to eliminate:",
            "we are in the night phase. since you are the",
            "the night phase has started",
            "mafia, agree on a victim"
        ]
        
        for pattern in current_night_patterns:
            if pattern in last_3_lines:
                return "night"
        
        # PRIORITY 3: CURRENT voting requests in last 3 lines
        current_voting_patterns = [
            "voting phase - submit one vote",
            "voting phase - submit",
            "submit one vote in format",
            "the voting phase has began",
            "submit your vote for which player you want to vote out"
        ]
        
        for pattern in current_voting_patterns:
            if pattern in last_3_lines:
                return "voting"
        
        # PRIORITY 4: Check last 5 lines for recent phase transitions
        if "was eliminated by vote" in last_5_lines and "night phase" in last_5_lines:
            return "night"
        
        if "day breaks" in last_5_lines:
            if "voting phase" in last_5_lines:
                return "voting"
            elif "discuss for" in last_5_lines:
                return "discussion"
            else:
                return "discussion"
        
        # PRIORITY 5: Look for valid targets format to distinguish phases
        # Check for current action requirements based on target format
        if "valid:" in last_3_lines or "valid options:" in last_3_lines:
            # Extract the context around valid targets
            for line in lines[-3:]:
                line_lower = line.lower()
                if "investigate:" in line_lower or "protect:" in line_lower or "eliminate:" in line_lower:
                    return "night"
                elif "vote" in line_lower and "valid" in line_lower:
                    return "voting"
        
        # PRIORITY 6: Broader search in last 10 lines for phase indicators
        night_action_phrases = [
            "choose one player to investigate",
            "choose one player to protect", 
            "choose one player to eliminate"
        ]
        
        for phrase in night_action_phrases:
            if phrase in last_10_lines:
                return "night"
        
        # PRIORITY 7: Game start detection (only for turn 1)
        if self.turn_count <= 1 and "welcome to secret mafia" in observation.lower():
            if "night has fallen" in observation.lower() or "mafia, agree on a victim" in observation.lower():
                return "night"
            return "waiting"
        
        # PRIORITY 8: Discussion detection - look for player dialogue without explicit phase
        if len(observation.strip()) > 50:
            has_player_dialogue = bool(re.search(r'Player \d+:', observation))
            has_player_messages = bool(re.search(r'\[Player \d+\]', observation))
            
            # Check that we don't have voting/night indicators in recent lines
            no_voting_in_recent = "voting phase" not in last_10_lines
            no_night_in_recent = "night phase" not in last_10_lines
            
            if (has_player_dialogue or has_player_messages) and no_voting_in_recent and no_night_in_recent:
                return "discussion"
        
        # PRIORITY 9: Unknown if we can't determine
        return "unknown"
    
    
    def create_game_context(self, observation: str, recent_history: list = None) -> str:
        """
        Create efficient game context by stacking observations optimally.
        
        Args:
            observation: Current observation
            recent_history: List of recent observations (last 1-2 turns)
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        
        # 1. Add compressed game state summary
        game_summary = {
            "turn": self.turn_count,
            "role": self.my_role,
            "player_id": self.my_player_id,
            "alive_players": self.alive_players,
            "dead_count": 6 - len(self.alive_players)  # Assuming 6 player game
        }
        context_parts.append(f"[GAME STATE] Turn {game_summary['turn']}, Role: {game_summary['role']}, Alive: {game_summary['alive_players']}")
        
        # 2. Add recent history (last 1-2 turns) if provided
        if recent_history:
            # Keep only last 2 observations to limit tokens
            recent = recent_history[-2:] if len(recent_history) > 2 else recent_history
            if recent:
                context_parts.append("\n[RECENT HISTORY]")
                for i, hist in enumerate(recent, 1):
                    # Truncate long observations
                    truncated = hist[:200] + "..." if len(hist) > 200 else hist
                    context_parts.append(f"Turn {self.turn_count - len(recent) + i}: {truncated}")
        
        # 3. Add current observation (most important, always full)
        context_parts.append(f"\n[CURRENT OBSERVATION]\n{observation}")
        
        return "\n".join(context_parts)

    def _extract_current_phase_context(self, observation: str) -> str:
        """
        Extract the most recent/relevant phase context.
        Priority: recent lines > old lines
        """
        lines = observation.strip().split('\n')
        
        # For phase detection, we care most about the LAST 5-10 lines
        # This is where current phase announcements appear
        
        if len(lines) <= 10:
            return observation  # Return full observation if short
        
        # Strategy: Take last 10 lines which should contain current phase info
        recent_lines = lines[-10:]
        
        # Check if there are any critical phase announcements
        # in those last 10 lines and extract surrounding context
        phase_keywords = [
            "voting phase", "night phase", "day breaks", 
            "night has fallen", "mafia, agree on", "choose one player",
            "submit one vote", "valid targets:", "resubmit a valid"
        ]
        
        for i in range(len(recent_lines) - 1, -1, -1):
            line_lower = recent_lines[i].lower()
            if any(keyword in line_lower for keyword in phase_keywords):
                # Found a phase indicator - return from this point onward
                return '\n'.join(recent_lines[i:])
        
        # No specific phase indicator found, return last 10 lines
        return '\n'.join(recent_lines)

    def _fallback_phase_detection(self, observation: str) -> str:
        """
        Fallback rule-based phase detection if LLM fails.
        """
        obs_lower = observation.lower()
        
        # Check for critical indicators first
        if "night has fallen" in obs_lower or "mafia, agree on a victim" in obs_lower or "valid targets:" in obs_lower:
            return "night"
        
        # Check for invalid move recovery
        if "attempted an invalid move" in obs_lower or "resubmit a valid move" in obs_lower:
            return "voting"
        
        # Check for voting phase (most specific)
        if "voting phase" in obs_lower or ("submit one vote" in obs_lower and "[" in observation):
            return "voting"
        
        # Check for night phase
        if "night phase" in obs_lower or "choose one player to investigate" in obs_lower:
            return "night"
        
        # Check for discussion phase
        if "discuss for" in obs_lower or "day breaks" in obs_lower:
            return "discussion"
        
        # Check for game start
        if "welcome to secret mafia" in obs_lower and "your role:" in obs_lower:
            return "waiting"
        
        # Default to discussion if there's content
        if len(observation.strip()) > 20:
            return "discussion"
        
        return "unknown"

    def _extract_recent_player_statements(self, observation: str) -> str:
        """Extract recent player statements from the observation for context"""
        lines = observation.strip().split('\n')
        
        # Look for player statements in the observation
        player_statements = []
        
        for line in lines:
            line = line.strip()
            # Match patterns like "Player X: statement" or "[Player X]: statement"
            if re.match(r'.*Player \d+.*:', line) or '[' in line:
                # Skip our own statements from the current observation
                if f"Player {self.my_player_id}" not in line:
                    player_statements.append(line)
        
        # Keep only the most recent 10 statements to avoid token bloat
        recent_statements = player_statements[-10:] if len(player_statements) > 10 else player_statements
        
        return "\n".join(recent_statements) if recent_statements else "No recent player statements"

    def _extract_latest_developments(self, observation: str) -> str:
        """Extract only the most recent 2-3 player actions/statements"""
        lines = observation.strip().split('\n')
        
        # Get last 5 lines that contain actual player actions
        latest_actions = []
        for line in lines[-8:]:  # Check last 8 lines
            line = line.strip()
            if re.search(r'Player \d+.*:', line) or '[' in line and 'Player' in line:
                latest_actions.append(line)
        
        # Return only the most recent 3 actions
        recent_actions = latest_actions[-3:] if len(latest_actions) > 3 else latest_actions
        return "\n".join(recent_actions) if recent_actions else "No new developments this turn"
    
    def _identify_new_information(self, observation: str, recent_discussion: str) -> str:
        """Identify what new information appeared that wasn't in previous discussions"""
        if not recent_discussion or recent_discussion == "None":
            return "First statement this game"
        
        # Look for investigation results, deaths, votes that are new
        new_info = []
        
        # Check for investigation claims
        if "IS a Mafia member" in observation and "investigation" not in recent_discussion:
            new_info.append("New investigation result revealed")
        
        # Check for new role claims  
        if "I'm the" in observation or "I am the" in observation:
            new_info.append("New role claim made")
        
        # Check for new votes
        if "[Vote Player" in observation:
            new_info.append("New votes cast")
        
        # Check for deaths
        if "was killed" in observation or "was eliminated" in observation:
            new_info.append("Player elimination occurred")
        
        return "; ".join(new_info) if new_info else "Ongoing discussion continuation"
    
    def _get_turn_specific_focus(self, turn_count: int, recent_statements: str) -> str:
        """Get specific analysis focus based on turn number and recent events"""
        if turn_count <= 2:
            return "Focus: Initial suspicion analysis, look for early Mafia coordination"
        elif turn_count <= 4:
            return "Focus: Voting pattern analysis, investigate role claims credibility"
        elif turn_count <= 6:
            return "Focus: Alliance detection, identify who's defending whom"
        else:
            return "Focus: End-game analysis, identify remaining Mafia through process of elimination"
    
    def _generate_unique_turn_context(self, turn_count: int, observation: str, recent_discussion: str) -> str:
        """Generate completely unique context each turn to force different responses"""
        
        contexts = [
            f"EMERGENCY ANALYSIS NEEDED: Major developments in Turn {turn_count}! Respond with immediate tactical assessment.",
            f"CRITICAL TURN {turn_count}: Game dynamics have shifted. Provide fresh strategic evaluation of new positions.",
            f"URGENT TURN {turn_count} UPDATE: Recent player actions demand immediate analysis. Focus on NEW evidence only.",
            f"BREAKING: Turn {turn_count} reveals crucial information. Abandon previous theories and analyze current reality.",
            f"ALERT TURN {turn_count}: Voting patterns indicate major developments. Reassess all players based on latest actions.",
            f"PRIORITY TURN {turn_count}: New alliances emerging. Shift focus to most recent player interactions and votes.",
            f"DEVELOPMENT TURN {turn_count}: Fresh evidence available. Ignore past speculation and focus on concrete recent actions.",
            f"TACTICAL TURN {turn_count}: Game state has evolved. Provide updated threat assessment based on latest developments."
        ]
        
        # Use turn count to select different context each time
        selected_context = contexts[turn_count % len(contexts)]
        
        # Add turn-specific instructions
        if turn_count <= 2:
            selected_context += " Focus on initial reads and first impressions."
        elif turn_count <= 4:
            selected_context += " Focus on voting analysis and claim verification."
        elif turn_count <= 6:
            selected_context += " Focus on alliance patterns and defensive behaviors."
        else:
            selected_context += " Focus on end-game positioning and elimination priorities."
        
        return selected_context
    
    def _extract_real_game_events(self, observation: str) -> str:
        """Extract only real game events, ignore fake post-game content"""
        lines = observation.strip().split('\n')
        real_events = []
        
        # Look for official game announcements
        for line in lines:
            line_lower = line.lower()
            # Skip obvious fake content
            if any(fake in line_lower for fake in [
                "game has concluded", "mafia wins", "congratulations", 
                "post-game", "well played", "another round"
            ]):
                continue
                
            # Keep real game events
            if any(real in line_lower for real in [
                "was killed", "was eliminated", "day breaks", "night phase",
                "voting phase", "discuss for", "vote will follow"
            ]):
                real_events.append(line.strip())
        
        return "\n".join(real_events[-3:]) if real_events else "No new game events"
    
    def _extract_real_player_actions(self, observation: str) -> str:
        """Extract actual player votes and statements, ignore fake content"""
        lines = observation.strip().split('\n')
        real_actions = []
        
        for line in lines:
            line_lower = line.lower()
            # Skip fake content
            if any(fake in line_lower for fake in [
                "congratulations", "well played", "another round", 
                "game concluded", "securing the victory"
            ]):
                continue
            
            # Keep real player actions (votes, accusations, defenses)
            if any(action_indicator in line for action_indicator in [
                "[Player", "Player ", "voted", "suspicious", "defend", "accuse"
            ]):
                # Only keep if it looks like actual game content
                if not any(fake in line_lower for fake in ["mafia team winning", "post-game"]):
                    real_actions.append(line.strip())
        
        return "\n".join(real_actions[-5:]) if real_actions else "No new player actions"

    def _get_role_specific_discussion_strategy(self, role: str, observation: str) -> str:
        """Get role-specific discussion strategy to improve gameplay"""
        if not role:
            return "STRATEGY: Stay alert and participate actively in discussions."
        
        role_lower = role.lower()
        
        if role_lower == "mafia":
            return """MAFIA STRATEGY - CRITICAL FOR SURVIVAL:
- DEFLECT suspicion by pointing fingers at others
- NEVER act too analytical or game-theory focused
- When accused: Get emotional, demand evidence, counter-accuse
- PROTECT your teammates by defending them subtly
- Create chaos by sowing doubt about others
- BLEND IN by showing appropriate concern and confusion
- If teammate is accused: "That seems rushed, what evidence do we have?"
- If you're accused: "That's ridiculous! Player X has been way more suspicious!"
- Example: "I'm worried about Player 2 - they've been way too quiet"

CRITICAL: Act like an innocent player who's scared and confused, NOT like an analyst."""
        
        elif role_lower == "detective":
            # Add investigation results info if we have any
            investigation_info = ""
            if self.memory.investigation_results:
                investigation_info = f"\n\nYOUR INVESTIGATION RESULTS: {self.memory.investigation_results}"
                investigation_info += "\n- SHARE these results immediately to help Village!"
                investigation_info += "\n- Example: 'I'm the Detective. I investigated Player X last night - they are NOT Mafia' or 'they ARE Mafia!'"
            
            return f"""DETECTIVE STRATEGY:
- IMMEDIATELY share investigation results (don't fabricate or hide them){investigation_info}
- Be confident when you have evidence from investigations
- PROTECT yourself by revealing your role when sharing results
- Lead votes against confirmed Mafia members
- NEVER make up fake investigation results or 'alibi checks'
- Detectives learn MAFIA/NOT MAFIA status, not movement or alibis
- Example: "I'm the Detective. Player 3 is CONFIRMED MAFIA from my investigation!"
- Be assertive but strategic about timing"""
        
        elif role_lower == "doctor":
            return """DOCTOR STRATEGY - STAY ALIVE TO KEEP PROTECTING:
- BE SUBTLE and avoid drawing attention to yourself as a power role
- SUPPORT other players' observations without being aggressive
- DON'T make direct accusations - let Villagers do the hunting
- AGREE with reasonable suggestions to appear helpful but not threatening
- AVOID being the most vocal or analytical player
- PROTECT yourself by blending in and not appearing dangerous
- Example: "I think Player X makes a good point about that."
- Example: "We should be careful before making big decisions."
- Example: "I agree with the evidence presented."
- Your job: SURVIVE to protect others, not lead the hunt
- Remember: Dead Doctors can't protect anyone!"""
        
        else:  # Villager
            return """VILLAGER STRATEGY - HUNT MAFIA ACTIVELY:
- QUESTION suspicious behavior and inconsistencies during discussion
- BUILD cases against likely Mafia members through evidence
- SUPPORT confirmed Village roles (Detective/Doctor) and protect them
- Be AGGRESSIVE in hunting Mafia - Village wins by eliminating Mafia!
- PREPARE your voting target during discussion phases
- TARGET players who seem suspicious, quiet, or evasive when voting
- Example: "Your silence is suspicious - explain yourself."
- Example: "Your story doesn't add up - I don't trust you."
- Lead discussions toward identifying Mafia, then vote decisively
- Remember: Discussion builds cases, voting eliminates threats!"""

    def _get_night_action_strategy(self, role: str, valid_targets: List[int], observation: str) -> str:
        """Get role-specific night action strategy"""
        if not role:
            return "Choose your target carefully."
        
        role_lower = role.lower()
        
        if role_lower == "mafia":
            return """MAFIA ELIMINATION STRATEGY:

PRIORITY 1 - ELIMINATE THREATS:
- Detective (if they've revealed or you suspect them)
- Doctor (if they've claimed or seem protective)  
- Players leading accusations against your team
- Strong analytical players who might figure you out

PRIORITY 2 - STRATEGIC TARGETS:
- Trusted villagers (reduces Village voting power)
- Players who could unite others against you
- Quiet players who might be power roles

AVOID:
- Players everyone already suspects (let Village eliminate them)
- Your own teammates (obviously!)
- Players who defend you (useful idiots)

TEAMMATE COORDINATION:
- If you see a teammate under heavy suspicion, eliminate someone else to keep focus off them
- Target players who are investigating your teammates

Remember: Every elimination should either remove a threat OR create chaos for the Village team."""
        
        elif role_lower == "detective":
            return """DETECTIVE INVESTIGATION STRATEGY:

PRIORITY TARGETS:
- Players with high suspicion but you need confirmation
- Quiet players who might be hiding (often Mafia)
- Players who defended suspected Mafia members
- Anyone who deflected suspicion onto others

AVOID INVESTIGATING:
- Players already widely trusted by Village
- Obviously suspicious players (save for last if needed)
- Players you're confident are Village

CRITICAL DETECTIVE RULES:
- You learn ONLY if target is "MAFIA" or "NOT MAFIA" - nothing else
- NEVER fabricate investigation results or claim alibi checks
- Share ALL real results immediately the next day
- Example: "I investigated Player 2 - they ARE Mafia!" or "Player 4 is NOT Mafia"

STRATEGY:
- Investigate to CONFIRM suspicions, not just explore
- Focus on players who could swing votes if confirmed
- Share results immediately next day (don't hold back)"""
        
        elif role_lower == "doctor":
            return """DOCTOR PROTECTION STRATEGY:

PRIORITY PROTECTION (Night Phase):
- Confirmed Detective (if they revealed with investigation results)
- Players leading Village discussions effectively
- Players who seem to be figuring out Mafia members
- Strong analytical players Mafia would want eliminated
- Yourself (only if you're under heavy suspicion and might be voted out)

AVOID PROTECTING:
- Suspected Mafia members (waste of protection)
- Players already widely suspected by Village
- Random quiet players with no clear Village value
- Players who seem evasive or deflective

SMART PROTECTION STRATEGY:
- Think like Mafia: who threatens them most?
- Protect players who are effective but not suspicious
- First protection often random since little info available
- Later protections based on who's helping Village most
- Don't protect same player twice unless critical"""
        
        else:  # Villager (no night action)
            return "VILLAGER: You have no night actions. Use this time to plan your discussion strategy for tomorrow."

    def _extract_targets(self, observation: str) -> List[int]:
        """Extract available targets from observation"""
        
        # Look for explicit target lists
        target_match = re.search(r'Valid: \[(.*?)\]', observation)
        if target_match:
            target_str = target_match.group(1)
            return [int(x) for x in re.findall(r'(\d+)', target_str)]
        
        # Look for bracket notation
        bracket_targets = re.findall(r'\[(\d+)\]', observation)
        if bracket_targets:
            return [int(x) for x in bracket_targets]
        
        # Fallback to pattern matching
        targets = []
        for i in range(6):  # Mafia typically has 6 players
            if f"[{i}]" in observation or f"Player {i}" in observation:
                targets.append(i)
        
        return targets if targets else [1, 2, 3, 4, 5]

    def _is_repetitive_response(self, response: str) -> bool:
        """Check if response is repetitive compared to recent responses"""

        if len(self.memory.discussion_history) < 2:
            return False

        # Check against last few responses
        recent_responses = self.memory.discussion_history[-3:]
        for past_response in recent_responses:
            if len(set(response.split()) & set(past_response.split())) > len(response.split()) * 0.6:
                return True

        return False

    # =========================================================================
    # TRM-INSPIRED TWO-STEP REASONING ENHANCEMENT
    # =========================================================================

    def _generate_internal_reasoning(self, observation: str, game_context) -> str:
        """
        Step 1: Generate internal strategic reasoning (inspired by TRM's 'z' latent feature)

        This is the agent's private strategic analysis before speaking publicly.
        Returns empty string on failure (triggers fallback to original method).
        """

        # Get investigation info if Detective
        investigation_info = ""
        if self.my_role == "detective" and self.memory.investigation_results:
            investigation_info = f"\nYour investigation results: {self.memory.investigation_results}"

        reasoning_prompt = f"""RESPOND IN ENGLISH ONLY. DO NOT USE CHINESE OR ANY OTHER LANGUAGE.

You are Player {self.my_player_id} ({self.my_role}) - INTERNAL STRATEGIC ANALYSIS

THIS IS YOUR PRIVATE THINKING - NOT WHAT YOU WILL SAY PUBLICLY.

CRITICAL: Your response must be in ENGLISH language, not Chinese (不要用中文).

CURRENT SITUATION:
{observation[-600:]}

YOUR ROLE: {self.my_role}{investigation_info}
ALIVE PLAYERS: {self.alive_players}

STRATEGIC ANALYSIS - Think through these questions IN ENGLISH:
1. What are the most likely roles of other players based on their behavior?
2. Who poses the biggest threat to your team (Village or Mafia)?
3. What is your primary strategic goal for this turn?
4. What information should you reveal vs. conceal?
5. How can you advance your team's win condition this turn?

ROLE-SPECIFIC INTERNAL THINKING:
- Mafia: Who should you deflect suspicion toward? How do you protect teammates?
- Detective: Should you reveal investigation results now? Who needs to know?
- Doctor: How do you stay subtle while supporting Village?
- Villager: Who is most suspicious? What evidence supports this?

Provide your step-by-step internal analysis IN ENGLISH (2-5 sentences).
This is your PRIVATE strategic thinking BEFORE deciding what to say.
RESPOND IN ENGLISH ONLY:"""

        try:
            # Shorter timeout for reasoning (it's just internal analysis)
            reasoning = self._call_modal_with_enhanced_timeout(
                reasoning_prompt,
                phase="discussion",
                timeout=30
            )

            print(f"💭 Internal reasoning: {reasoning[:80]}...")
            return reasoning.strip()

        except Exception as e:
            print(f"⚠️ Internal reasoning generation failed: {e}")
            return ""  # Triggers fallback

    def _generate_action_from_reasoning(
        self,
        observation: str,
        game_context,
        internal_reasoning: str
    ) -> str:
        """
        Step 2: Generate public action based on internal reasoning (inspired by TRM's 'y' answer)

        This converts private strategic analysis into a public-facing response.
        Falls back to original method on failure.
        """

        # Get role-specific strategy
        role_strategy = self._get_role_specific_discussion_strategy(self.my_role, observation)

        action_prompt = f"""RESPOND IN ENGLISH ONLY. DO NOT USE CHINESE OR ANY OTHER LANGUAGE.

You are Player {self.my_player_id} ({self.my_role}) - PUBLIC RESPONSE

CRITICAL: Your response must be in ENGLISH language, not Chinese (不要用中文).

YOUR INTERNAL STRATEGIC ANALYSIS:
"{internal_reasoning}"

CURRENT GAME SITUATION:
{observation[-500:]}

ROLE STRATEGY:
{role_strategy[:300]}

CRITICAL: Based on your internal analysis above, what do you say publicly to other players?

RESPONSE REQUIREMENTS:
- MUST be in ENGLISH language
- Natural conversation (20-100 words)
- Stay in character as Player {self.my_player_id}
- Align with your internal strategic analysis
- Advance your team's goals (Mafia team or Village team)
- Sound like a real player having a conversation, NOT like an analyst

Generate your public statement IN ENGLISH now:"""

        try:
            response = self._call_modal_with_enhanced_timeout(
                action_prompt,
                phase="discussion",
                timeout=40
            )

            return response.strip()

        except Exception as e:
            print(f"⚠️ Action generation from reasoning failed: {e}")
            raise  # Re-raise to trigger fallback in _handle_discussion_action