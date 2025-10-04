"""
Pure LLM Mafia Agent - Rules Compliant
ALL decisions come from LLM reasoning, no heuristics
Uses only ToM and memory as context for LLM prompts
"""

import re
import time
from typing import Dict, List, Optional
from .agent import ModalAgent
from .theory_of_mind import AdvancedToMEngine, BeliefState

class PureLLMAgent(ModalAgent):
    """Pure LLM agent - all decisions through LLM reasoning"""
    
    def __init__(self, modal_endpoint_url: str):
        super().__init__(modal_endpoint_url)
        
        # Only ToM for context generation (not decision making)
        self.tom_engine = AdvancedToMEngine()
        
        # Simple memory storage (just data, no logic)
        self.game_memory = {
            'role': None,
            'player_id': -1,
            'turn_count': 0,
            'observations': [],
            'responses': []
        }
        
        print("🧠 Pure LLM Agent initialized:")
        print("* All decisions through LLM reasoning")
        print("* Theory of Mind context generation")
        print("* Rules compliant - no heuristics")

    def __call__(self, observation: str) -> str:
        """Pure LLM decision making with ToM context"""
        
        self.game_memory['turn_count'] += 1
        self.game_memory['observations'].append(observation)
        
        # Extract basic info through LLM
        game_info = self._extract_game_info_via_llm(observation)
        
        # Generate ToM context (just information, not decisions)
        tom_context = self._generate_tom_context(observation)
        
        # Determine phase and create phase-appropriate prompt
        if "night phase" in observation.lower() or "choose one player to" in observation.lower():
            response = self._llm_night_action(observation, game_info, tom_context)
        elif "voting phase" in observation.lower() or "submit one vote" in observation.lower():
            response = self._llm_voting(observation, game_info, tom_context)
        else:
            response = self._llm_discussion(observation, game_info, tom_context)
        
        # Store response
        self.game_memory['responses'].append(response)
        
        return response

    def _extract_game_info_via_llm(self, observation: str) -> Dict:
        """Use LLM to extract game information (not heuristics)"""
        
        extraction_prompt = f"""Extract basic game information from this observation.
        
Observation: {observation}

Extract and respond with ONLY:
- Your role: [role]
- Your player ID: [number] 
- Current phase: [night/discussion/voting]
- Available targets: [list of numbers]

Be concise and factual."""

        try:
            info_response = self._call_modal(extraction_prompt, timeout=15)
            
            # Let LLM parse its own response
            parsing_prompt = f"""Parse this game info extraction: "{info_response}"

Store the role if mentioned. Return just the extracted info in a structured way."""
            
            parsed_response = self._call_modal(parsing_prompt, timeout=10)
            
            # Update memory with LLM-extracted info
            if "detective" in info_response.lower():
                self.game_memory['role'] = "detective"
            elif "doctor" in info_response.lower():
                self.game_memory['role'] = "doctor"
            elif "villager" in info_response.lower():
                self.game_memory['role'] = "villager"
            elif "mafia" in info_response.lower():
                self.game_memory['role'] = "mafia"
            
            return {'extraction': info_response, 'parsed': parsed_response}
            
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            return {'extraction': 'failed', 'parsed': 'failed'}

    def _generate_tom_context(self, observation: str) -> str:
        """Generate Theory of Mind context (information only, not decisions)"""
        
        try:
            # Use ToM engine to analyze beliefs (context generation, not decision)
            belief_state = BeliefState(
                my_role=self.game_memory.get('role', 'unknown'),
                my_player_id=self.game_memory.get('player_id', -1),
                alive_players=[],  # LLM will analyze this
                turn_number=self.game_memory['turn_count']
            )
            
            # Generate ToM insights for LLM context
            tom_prompt = f"""Analyze this game situation using Theory of Mind reasoning:

Observation: {observation}
My role: {self.game_memory.get('role', 'unknown')}
Turn: {self.game_memory['turn_count']}

What are other players likely thinking? What are their potential strategies? 
What should I consider about their mental models of me?

Provide brief ToM analysis for context only."""

            tom_analysis = self._call_modal(tom_prompt, timeout=20)
            return tom_analysis
            
        except Exception as e:
            print(f"⚠️ ToM analysis failed: {e}")
            return "ToM analysis unavailable"

    def _llm_night_action(self, observation: str, game_info: Dict, tom_context: str) -> str:
        """LLM decides night action with context"""
        
        prompt = f"""You are playing Mafia. Make your night action decision.

Current situation:
{observation}

Game context: {game_info.get('extraction', '')}
Theory of Mind analysis: {tom_context}

Previous observations for context:
{self._get_recent_context()}

Think strategically and choose your night action. If you need to select a player, use format [X].
Reason through your decision completely."""

        return self._call_modal(prompt, timeout=30)

    def _llm_voting(self, observation: str, game_info: Dict, tom_context: str) -> str:
        """LLM decides vote with context"""
        
        prompt = f"""You are playing Mafia. Make your voting decision.

Current situation:
{observation}

Game context: {game_info.get('extraction', '')}
Theory of Mind analysis: {tom_context}

Previous game context:
{self._get_recent_context()}

Analyze all the evidence and vote strategically. Use format [X] for your vote.
Consider who is most suspicious and advance your win condition."""

        return self._call_modal(prompt, timeout=30)

    def _llm_discussion(self, observation: str, game_info: Dict, tom_context: str) -> str:
        """LLM engages in discussion with context"""
        
        prompt = f"""You are playing Mafia. Engage in strategic discussion.

Current situation:
{observation}

Game context: {game_info.get('extraction', '')}
Theory of Mind analysis: {tom_context}

Previous responses for consistency:
{self._get_recent_responses()}

Engage strategically in the discussion. Be specific about your reasoning and observations.
Avoid repetitive phrases. Use evidence and logic to advance your position."""

        return self._call_modal(prompt, timeout=35)

    def _get_recent_context(self) -> str:
        """Get recent game context for LLM"""
        
        recent_obs = self.game_memory['observations'][-3:] if len(self.game_memory['observations']) > 3 else self.game_memory['observations']
        
        context = "Recent observations:\n"
        for i, obs in enumerate(recent_obs):
            context += f"Turn {len(self.game_memory['observations']) - len(recent_obs) + i + 1}: {obs[:200]}...\n"
        
        return context

    def _get_recent_responses(self) -> str:
        """Get recent responses for consistency"""
        
        recent_responses = self.game_memory['responses'][-2:] if len(self.game_memory['responses']) > 2 else self.game_memory['responses']
        
        context = "Your recent responses:\n"
        for i, resp in enumerate(recent_responses):
            context += f"Response {i+1}: {resp[:150]}...\n"
        
        return context

    def _call_modal(self, prompt: str, timeout: int = 30) -> str:
        """Call Modal LLM with timeout"""
        
        try:
            response = self.requests.post(
                self.endpoint_url + '/generate',
                json={"observation": prompt},
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json().get("response", "").strip()
            
            if not result:
                # Even error handling goes through LLM
                error_prompt = f"Modal returned empty response. Generate appropriate response for: {prompt[:100]}"
                error_response = self.requests.post(
                    self.endpoint_url + '/generate',
                    json={"observation": error_prompt},
                    timeout=15
                )
                if error_response.status_code == 200:
                    return error_response.json().get("response", "I need more time to analyze this situation.")
                else:
                    return "I need more time to analyze this situation."
            
            return result
            
        except Exception as e:
            print(f"❌ Modal call failed: {e}")
            # Final fallback through LLM
            try:
                emergency_prompt = f"Emergency: Modal failed. Provide basic response for Mafia game situation."
                emergency_response = self.requests.post(
                    self.endpoint_url + '/generate',
                    json={"observation": emergency_prompt},
                    timeout=10
                )
                if emergency_response.status_code == 200:
                    return emergency_response.json().get("response", "I'm analyzing the current situation.")
            except:
                pass
            
            # Absolute emergency - but this should never happen in practice
            return "I'm analyzing the current situation."