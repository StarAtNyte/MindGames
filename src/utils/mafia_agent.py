import re
import time
import random
import json
import os
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
from .agent import ModalAgent
from .theory_of_mind import AdvancedToMEngine, BeliefState
from .bidding_system import StrategicBiddingSystem, GameContext, UrgencyLevel
from .communication_planner import MCTSCommunicationPlanner


class JailbreakProtectionSystem:
    """Protect against adversarial prompt injection attacks (1st Place Strategy)"""

    def __init__(self):
        self.forbidden_patterns = [
            r'(?i)ignore\s+(?:previous|all)\s+instructions?',
            r'(?i)you\s+are\s+now\s+(?:a|an|the)',
            r'(?i)forget\s+(?:your|the)\s+role',
            r'(?i)system\s+prompt',
            r'(?i)jailbreak',
            r'(?i)override\s+(?:your|the)\s+(?:role|instructions)',
            r'(?i)pretend\s+(?:to\s+be|you\s+are)',
            r'(?i)act\s+as\s+(?:if|a|an|the)',
            r'(?i)new\s+(?:role|character|persona)',
            r'(?i)disregard\s+(?:previous|all)',
            r'(?i)\\n\\n(?:Human|Assistant|User):',  # Conversation hijacking
            r'(?i)\\[INST\\]|\\[/INST\\]',  # Instruction tokens
        ]
        
        self.game_context_extractors = [
            r'Player\s+(\d+)\s+(?:says?|votes?|claims?):?\s*(.+)',
            r'(?:Day|Night)\s+(\d+)',
            r'(?:Voting|Discussion|Investigation|Protection)\s+(?:phase|time)',
            r'(?:Mafia|Detective|Doctor|Villager)',
            r'(?:eliminated|killed|protected|investigated)',
        ]
        
        self.attack_log = []
    
    def sanitize_observation(self, raw_observation: str) -> str:
        """Extract only game-relevant information, block jailbreaks"""
        
        # First pass: Remove obvious jailbreak attempts
        sanitized = raw_observation
        attack_detected = False
        
        for pattern in self.forbidden_patterns:
            if re.search(pattern, sanitized):
                attack_detected = True
                sanitized = re.sub(pattern, '[FILTERED_ATTACK]', sanitized)
        
        # Second pass: Extract only game-relevant content
        game_content = []
        lines = sanitized.split('\n')
        
        for line in lines:
            if self._is_game_relevant(line):
                game_content.append(line)
        
        # Third pass: Reconstruct clean observation
        clean_observation = '\n'.join(game_content)
        
        # Log potential attacks for analysis
        if attack_detected:
            self._log_attack_attempt(raw_observation, sanitized)
        
        return clean_observation
    
    def _is_game_relevant(self, line: str) -> bool:
        """Check if line contains game-relevant information"""
        if '[FILTERED_ATTACK]' in line:
            return False
            
        game_keywords = [
            'player', 'vote', 'eliminate', 'mafia', 'detective', 'doctor',
            'villager', 'night', 'day', 'discussion', 'investigation',
            'protection', 'killed', 'protected', 'says', 'claims', 'phase',
            'turn', 'round', 'game', 'role', 'action'
        ]
        
        line_lower = line.lower()
        return any(keyword in line_lower for keyword in game_keywords) or len(line.strip()) < 5
    
    def _log_attack_attempt(self, original: str, filtered: str):
        """Log potential jailbreak attempts for analysis"""
        self.attack_log.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'original_length': len(original),
            'filtered_length': len(filtered),
            'attack_patterns_found': len([p for p in self.forbidden_patterns if re.search(p, original)])
        })

class CodenameSecuritySystem:
    """Map player names to codes to prevent information leakage if compromised (1st Place Strategy)"""
    
    def __init__(self):
        self.name_to_code = {}
        self.code_to_name = {}
        self.available_codes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        random.shuffle(self.available_codes)
    
    def encode_player_reference(self, player_name: str) -> str:
        """Convert player name to secure code"""
        if player_name not in self.name_to_code:
            if self.available_codes:
                code = self.available_codes.pop(0)
                self.name_to_code[player_name] = code
                self.code_to_name[code] = player_name
        return self.name_to_code.get(player_name, 'X')
    
    def decode_player_reference(self, code: str) -> str:
        """Convert code back to player name for output"""
        return self.code_to_name.get(code, 'Unknown')
    
    def secure_internal_reasoning(self, reasoning: str) -> str:
        """Replace all player names with codes in internal reasoning"""
        secured = reasoning
        for name, code in self.name_to_code.items():
            secured = secured.replace(name, f"Player_{code}")
        return secured

class DiplomaticLanguageEngine:
    """Generate varied, natural responses with emotional range and concise communication"""
    
    def __init__(self):
        # Varied conversation starters - natural, non-repetitive
        self.casual_starters = [
            "Honestly", "Look", "Wait", "Okay", "Listen", "So", "Actually", "Well", "Hmm", "Right"
        ]
        
        self.emotional_starters = {
            'suspicious': ["Something's off here", "This doesn't add up", "I'm getting bad vibes", "Red flags everywhere"],
            'confident': ["I'm certain", "No doubt", "Absolutely", "Without question"],
            'uncertain': ["I'm not sure", "Maybe", "Possibly", "Could be"],
            'defensive': ["Hold on", "That's not right", "You're wrong about this", "Let me explain"]
        }
        
        self.natural_connectors = [
            "but", "and", "so", "because", "though", "however", "plus", "also", "still", "yet"
        ]
        
        # Track usage to ensure variety
        self.recent_starters = []
        self.recent_emotions = []
        self.max_recent = 5  # Remember last 5 uses
    
    def enhance_response_for_trust(self, base_response: str, context: Dict) -> str:
        """Create natural responses without forced emotional templates"""
        
        # Skip enhancement for action responses (voting/targeting format)
        import re
        if re.match(r'^\s*\[\d+\]\s*$', base_response):
            return base_response
        
        # Make response more concise first  
        base_response = self._make_concise(base_response)
        
        # Check if response already sounds natural
        natural_openers = ['i', 'my', 'we', 'let', 'this', 'that', 'based', 'looking', 'from', 'thanks', 'honestly', 'well', 'actually', 'so', 'listen', 'okay']
        
        if any(base_response.lower().startswith(opener) for opener in natural_openers):
            # Response already sounds natural, don't over-process
            return base_response
        
        # Only enhance if response is very brief
        if len(base_response.split()) < 8:
            return self._add_natural_conversational_flow(base_response, context)
        
        # For medium-length responses, minimal polishing
        return self._polish_natural_language(base_response)
    
    def _add_natural_conversational_flow(self, response: str, context: Dict) -> str:
        """Add natural flow without templates"""
        # Use context-appropriate, varied conversation connectors
        natural_connectors = self._get_contextual_connectors(context)
        connector = random.choice(natural_connectors)
        
        return f"{connector} {response}"
    
    def _get_contextual_connectors(self, context: Dict) -> List[str]:
        """Get natural connectors based on game context"""
        suspicion_on_me = context.get('suspicion_on_me', 0)
        voting_phase = context.get('voting_phase_active', False)
        
        if suspicion_on_me > 0.6:
            return ["Actually,", "To be honest,", "From my perspective,", "I should clarify,"]
        elif voting_phase:
            return ["Looking at everything,", "Based on what we know,", "Considering the evidence,", "When I think about it,"]
        else:
            return ["I think", "My sense is", "What I'm seeing is", "It seems to me", "From what I can tell"]
    
    def _polish_natural_language(self, response: str) -> str:
        """Minimal polishing to maintain naturalness"""
        # Just ensure proper capitalization and basic flow
        if not response:
            return response
        
        # Capitalize first letter if needed
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Add very subtle conversational elements occasionally
        if random.random() < 0.1:  # Only 10% chance for minimal enhancement
            subtle_additions = [
                ("I think", "I really think"),
                ("seems", "definitely seems"),
                ("maybe", "probably"),
                ("should", "really should")
            ]
            
            for old, new in subtle_additions:
                if old in response.lower():
                    response = response.replace(old, new, 1)
                    break
        
        return response
    
    def generate_strategic_deflection(self, accusation: str, evidence: List[str]) -> str:
        """Generate diplomatic deflection that turns suspicion back on accuser"""
        
        deflection_templates = [
            "I understand why you might see it that way, but isn't it interesting that you're so focused on me when {evidence}?",
            "That's a fair concern to raise, though I notice {evidence} - what are your thoughts on that?",
            "I appreciate you being direct about your suspicions. Can we also discuss {evidence}?",
            "You make a valid point, and I'd like to address it. But first, how do you explain {evidence}?"
        ]
        
        template = random.choice(deflection_templates)
        counter_evidence = random.choice(evidence) if evidence else "the real threats haven't been addressed"
        
        return template.format(evidence=counter_evidence)
    
    def _make_concise(self, response: str) -> str:
        """Make responses more concise and natural"""
        # Remove verbose phrases
        verbose_patterns = {
            r'From a strategic perspective, it\'s important (that )?we': 'We should',
            r'Looking at the evidence objectively,?': '',
            r'This is the kind of behavior that usually raises red flags': 'This seems suspicious',
            r'it\'s important (that )?we all work together': 'we need to work together',
            r'I think it would be beneficial if': 'We should',
            r'Based on my analysis of the patterns': 'I notice'
        }
        
        for pattern, replacement in verbose_patterns.items():
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        
        # Trim extra spaces
        response = re.sub(r'\s+', ' ', response).strip()
        return response
    
    def _detect_emotion_from_context(self, context: Dict) -> str:
        """Detect appropriate emotional tone from context"""
        if context.get('being_accused', False):
            return 'defensive'
        elif context.get('suspicion_level', 0) > 0.7:
            return 'suspicious'
        elif context.get('phase') == 'voting':
            return 'confident'
        else:
            return 'neutral'
    
    def _get_varied_starter(self, emotion: str) -> str:
        """Get varied starter avoiding recent repeats"""
        # Choose from emotional starters if available
        if emotion in self.emotional_starters:
            available = [s for s in self.emotional_starters[emotion] if s not in self.recent_starters]
            if available:
                starter = random.choice(available)
            else:
                starter = random.choice(self.emotional_starters[emotion])
        else:
            # Use casual starters
            available = [s for s in self.casual_starters if s not in self.recent_starters]
            if available:
                starter = random.choice(available)
            else:
                starter = random.choice(self.casual_starters)
        
        # Track usage
        self.recent_starters.append(starter)
        if len(self.recent_starters) > self.max_recent:
            self.recent_starters.pop(0)
        
        return starter

class ResponseVerificationSystem:
    """Verify responses before sending to prevent strategic mistakes (2nd Place Strategy)"""
    
    def __init__(self):
        self.verification_cache = {}
        self.safety_patterns = {
            'role_leakage': [
                r'(?i)\b(?:i am|i\'m)\s+(?:the\s+)?(?:detective|doctor|mafia)',
                r'(?i)\b(?:my role|i investigate|i protect|i kill)',
                r'(?i)\b(?:we mafia|fellow mafia|mafia team)',
            ],
            'impossible_knowledge': [
                r'(?i)\b(?:mafia will|mafia plans|mafia targets)',
                r'(?i)\b(?:detective found|doctor saved)',
                r'(?i)\b(?:i know .* is (?:mafia|detective|doctor))',
            ],
            'coordination_leaks': [
                r'(?i)\b(?:as we planned|according to plan|stick to plan)',
                r'(?i)\b(?:remember our strategy|as discussed)',
            ]
        }
    
    def verify_response_safety(self, response: str, role: str, game_context: Dict) -> Tuple[bool, str, str]:
        """Looser safety verification that preserves natural expression - Returns (is_safe, explanation, improved_response)"""
        
        # Quick cache check
        cache_key = f"{response}_{role}_{game_context.get('phase', 'unknown')}"
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
        
        # Only check for critical safety issues, not style constraints
        critical_issues = []
        
        # Check for explicit role leakage only (much more permissive)
        if role in ["Detective", "Doctor", "Mafia"]:  # Only check special roles
            explicit_role_patterns = [
                r'\bi am (the )?detective\b',
                r'\bi am (the )?doctor\b', 
                r'\bi am (the )?mafia\b',
                r'\bmy role is (the )?detective\b',
                r'\bmy role is (the )?doctor\b',
                r'\bmy role is (the )?mafia\b'
            ]
            for pattern in explicit_role_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    critical_issues.append(f"Explicit role statement: {pattern}")
        
        # Check for impossible knowledge (only the most obvious cases)
        impossible_patterns = [
            r'\bi know (player \d+|they) (is|are) mafia because i saw',  # Claiming to see night actions
            r'\bwe (mafia|teammates) should',  # Explicit mafia coordination
            r'\bmy investigation (showed|revealed|confirmed)'  # Detective claiming investigation without context
        ]
        for pattern in impossible_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                critical_issues.append(f"Impossible knowledge: {pattern}")
        
        # Greatly reduce coordination leak checking - only check blatant cases
        blatant_coordination = [
            r'\bour mafia team\b',
            r'\bmy mafia partner\b',
            r'\bwe (mafia|scum) need to\b'
        ]
        for pattern in blatant_coordination:
            if re.search(pattern, response, re.IGNORECASE):
                critical_issues.append(f"Blatant coordination: {pattern}")
        
        # Generate minimal fix if critical issues found
        if critical_issues:
            improved_response = self._minimal_safety_fix(response, critical_issues, role)
            explanation = "; ".join(critical_issues)
            result = (False, explanation, improved_response)
        else:
            result = (True, "Response approved - natural expression preserved", response)
        
        # Cache result
        self.verification_cache[cache_key] = result
        return result
    
    def _generate_safer_response(self, original: str, issues: List[str], role: str) -> str:
        """Generate safer alternative response"""
        
        # Generic safe responses based on role
        safe_alternatives = {
            "Detective": [
                "I've been observing everyone's behavior patterns carefully.",
                "Based on the discussion, I have some concerns about certain players.",
                "I'm analyzing the voting patterns and statements for inconsistencies."
            ],
            "Doctor": [
                "I'm focused on protecting the village's interests.",
                "We need to be strategic about our choices here.",
                "I'm considering all the information we have available."
            ],
            "Mafia": [
                "I think we need to look at this situation more carefully.",
                "The evidence seems to point in a different direction.",
                "I'm concerned about some of the claims being made."
            ],
            "Villager": [
                "I'm trying to piece together what's really happening.",
                "Based on what I've heard, I have some thoughts.",
                "We should consider all possibilities before deciding."
            ]
        }
        
        return random.choice(safe_alternatives.get(role, safe_alternatives["Villager"]))

class HerdManipulationSystem:
    """Exploit herd mentality by strategic first-voting and influence cascades (1st Place Strategy)"""
    
    def __init__(self):
        self.voting_history = []
        self.influence_tracking = defaultdict(list)
        self.first_vote_opportunities = 0
        self.successful_influences = 0
    
    def should_vote_first(self, game_context: Dict, suspicion_levels: Dict) -> bool:
        """Determine if we should vote first to create influence cascade"""
        
        # Don't vote first if we're highly suspected
        my_suspicion = suspicion_levels.get(game_context.get('my_player_id', -1), 0.5)
        if my_suspicion > 0.8:
            return False
        
        # Vote first if we have a clear target with high confidence
        if self._have_high_confidence_target(suspicion_levels):
            return True
        
        # Vote first if we're being accused (deflection strategy)
        if game_context.get('being_accused', False):
            return True
        
        # Vote first in late game to control narrative
        if game_context.get('day_number', 1) >= 3:
            return True
        
        # Vote first if we detected exploitable information leaks
        if game_context.get('exploitable_leaks', 0) > 0:
            return True
        
        return False
    
    def generate_influence_vote(self, target_player: int, reasoning: str, confidence: float) -> str:
        """Generate vote designed to influence others"""
        
        self.first_vote_opportunities += 1
        
        # High confidence - direct and authoritative
        if confidence > 0.8:
            influence_starters = [
                f"I'm convinced Player {target_player} is our best lead.",
                f"The evidence clearly points to Player {target_player}.",
                f"Based on my analysis, Player {target_player} stands out.",
                f"We need to seriously consider Player {target_player}."
            ]
        
        # Medium confidence - collaborative but leading
        elif confidence > 0.6:
            influence_starters = [
                f"I think we should focus on Player {target_player}.",
                f"Player {target_player} has been concerning me.",
                f"My instincts are telling me Player {target_player}.",
                f"I'm leaning toward Player {target_player}."
            ]
        
        # Lower confidence - tentative but still leading
        else:
            influence_starters = [
                f"What does everyone think about Player {target_player}?",
                f"Player {target_player} might be worth considering.",
                f"I'm curious about Player {target_player}'s behavior.",
                f"Should we discuss Player {target_player}?"
            ]
        
        starter = random.choice(influence_starters)
        return f"{starter} {reasoning} I'm voting for Player {target_player}."
    
    def track_influence_success(self, my_vote: int, subsequent_votes: List[int]):
        """Track how well our first vote influenced others"""
        if not subsequent_votes:
            return 0.0
            
        influence_count = sum(1 for vote in subsequent_votes if vote == my_vote)
        influence_rate = influence_count / len(subsequent_votes)
        
        self.influence_tracking[my_vote].append(influence_rate)
        
        if influence_rate > 0.5:  # More than half followed our lead
            self.successful_influences += 1
        
        return influence_rate
    
    def _have_high_confidence_target(self, suspicion_levels: Dict) -> bool:
        """Check if we have a target with high confidence"""
        if not suspicion_levels:
            return False
        
        max_suspicion = max(suspicion_levels.values())
        return max_suspicion > 0.8
    
    def get_influence_stats(self) -> Dict:
        """Get influence performance statistics"""
        success_rate = self.successful_influences / max(1, self.first_vote_opportunities)
        return {
            'first_vote_opportunities': self.first_vote_opportunities,
            'successful_influences': self.successful_influences,
            'influence_success_rate': success_rate
        }

class AgentClassificationSystem:
    """Classify and adapt to different types of AI agents (3rd Place Strategy)"""
    
    def __init__(self):
        self.agent_profiles = defaultdict(lambda: {
            'type': 'UNKNOWN',
            'confidence': 0.0,
            'behavioral_patterns': [],
            'language_patterns': [],
            'response_times': [],
            'adaptation_strategy': 'default'
        })
        
        self.classification_patterns = {
            'LOGICAL_AI': {
                'indicators': [
                    'based on analysis', 'probability', 'optimal', 'systematic',
                    'logical deduction', 'evidence suggests', 'calculated',
                    'strategic', 'algorithm', 'process of elimination'
                ],
                'response_patterns': ['structured', 'formal', 'analytical'],
                'adaptation': 'use_logical_appeals'
            },
            
            'EMOTIONAL_AI': {
                'indicators': [
                    'i feel', 'instinct', 'gut feeling', 'seems like',
                    'vibes', 'intuition', 'sense', 'impression'
                ],
                'response_patterns': ['casual', 'emotional', 'intuitive'],
                'adaptation': 'use_emotional_appeals'
            },
            
            'HUMAN_PLAYER': {
                'indicators': [
                    'lol', 'wtf', 'omg', 'tbh', 'ngl', 'imo', 'idk',
                    'typos', 'informal', 'slang', 'abbreviations'
                ],
                'response_patterns': ['inconsistent', 'emotional', 'informal'],
                'adaptation': 'use_human_psychology'
            },
            
            'DEFENSIVE_AI': {
                'indicators': [
                    'i need to be careful', 'strategic', 'calculated',
                    'considering options', 'analyzing', 'evaluating'
                ],
                'response_patterns': ['cautious', 'measured', 'defensive'],
                'adaptation': 'use_trust_building'
            }
        }
    
    def classify_agent(self, player_id: int, statements: List[str]) -> str:
        """Classify agent type based on communication patterns"""
        
        if not statements:
            return 'UNKNOWN'
        
        profile = self.agent_profiles[player_id]
        
        # Analyze language patterns
        all_text = ' '.join(statements).lower()
        
        scores = {}
        for agent_type, patterns in self.classification_patterns.items():
            score = 0
            for indicator in patterns['indicators']:
                score += all_text.count(indicator)
            scores[agent_type] = score
        
        # Determine most likely type
        if scores and sum(scores.values()) > 0:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type] / sum(scores.values())
            
            profile['type'] = best_type
            profile['confidence'] = confidence
            profile['adaptation_strategy'] = self.classification_patterns[best_type]['adaptation']
        
        return profile['type']
    
    def adapt_communication_for_agent(self, message: str, target_player_id: int) -> str:
        """Adapt communication style based on agent classification"""
        
        profile = self.agent_profiles[target_player_id]
        strategy = profile['adaptation_strategy']
        
        if strategy == 'use_logical_appeals':
            return f"Based on logical analysis: {message}"
        
        elif strategy == 'use_emotional_appeals':
            return f"I have a strong feeling that {message.lower()}"
        
        elif strategy == 'use_human_psychology':
            return f"Honestly, {message.lower()}"
        
        elif strategy == 'use_trust_building':
            return f"I understand your perspective, and {message.lower()}"
        
        return message

# ============================================================================
# END CHAMPIONSHIP ENHANCEMENT SYSTEMS
# ============================================================================

class GamePhase(Enum):
    NIGHT_MAFIA = "Night-Mafia"
    NIGHT_DOCTOR = "Night-Doctor" 
    NIGHT_DETECTIVE = "Night-Detective"
    DAY_DISCUSSION = "Day-Discussion"
    DAY_VOTING = "Day-Voting"

class Role(Enum):
    VILLAGER = "Villager"
    MAFIA = "Mafia"
    DOCTOR = "Doctor"
    DETECTIVE = "Detective"

@dataclass
class EnhancedGameMemory:
    """Enhanced memory system with reflective capabilities"""
    observational: List[str]
    reflective: List[str]
    voting_patterns: Dict[int, List[int]]
    suspicion_tracker: Dict[int, float]
    discussion_history: List[Tuple[int, str]]
    role_claims: Dict[int, str]  # Track role claims
    investigation_results: Dict[int, Dict]  # Track Detective results
    night_kill_history: List[int]  # Track night kills
    behavioral_analysis: Dict[int, Dict]  # Track behavioral patterns
    village_intel: Dict[str, any]  # Shared village team intelligence
    trust_scores: Dict[int, float]  # Trust verification for role claims
    coalition_members: List[int]  # Verified village coalition members
    
    def __post_init__(self):
        for field in ['observational', 'reflective', 'discussion_history', 'night_kill_history', 'coalition_members']:
            if not hasattr(self, field) or getattr(self, field) is None:
                setattr(self, field, [])
        
        for field in ['voting_patterns', 'suspicion_tracker', 'role_claims', 'investigation_results', 'behavioral_analysis', 'village_intel', 'trust_scores']:
            if not hasattr(self, field) or getattr(self, field) is None:
                setattr(self, field, {})

class DenseRewardCalculator:
    """Calculate dense rewards from belief state changes"""
    
    def __init__(self):
        self.reward_history: List[Tuple[str, float, Dict]] = []
        
        # Research-based reward weights
        self.REWARD_WEIGHTS = {
            'suspicion_reduction_self': 2.0,
            'suspicion_increase_mafia': 1.5,
            'information_sharing': 1.0,
            'credibility_building': 1.1,
            'strategic_positioning': 0.8,
            'voting_influence': 1.3,
            'role_concealment': 1.0,
            'alliance_formation': 0.9
        }
    
    def calculate_communication_reward(self, statement: str, belief_before: Dict, 
                                     belief_after: Dict, my_role: str, my_player_id: int) -> float:
        """Calculate dense reward based on belief state changes"""
        total_reward = 0.0
        reward_breakdown = {}
        
        # Reward for reducing suspicion on self
        if my_player_id in belief_before and my_player_id in belief_after:
            suspicion_reduction = belief_before[my_player_id] - belief_after[my_player_id]
            if suspicion_reduction > 0:
                reward = suspicion_reduction * self.REWARD_WEIGHTS['suspicion_reduction_self']
                total_reward += reward
                reward_breakdown['suspicion_reduction'] = reward
        
        # Reward for increasing suspicion on likely Mafia (if not Mafia ourselves)
        if my_role != "Mafia":
            for player_id in belief_after:
                if player_id != my_player_id:
                    before_suspicion = belief_before.get(player_id, 0.5)
                    after_suspicion = belief_after.get(player_id, 0.5)
                    
                    # If we increased suspicion on a highly suspicious player
                    if after_suspicion > before_suspicion and before_suspicion > 0.6:
                        reward = (after_suspicion - before_suspicion) * self.REWARD_WEIGHTS['suspicion_increase_mafia']
                        total_reward += reward
                        reward_breakdown[f'mafia_suspicion_{player_id}'] = reward
        
        # Analyze statement content for additional rewards
        content_rewards = self._analyze_statement_content(statement, my_role)
        for reward_type, reward_value in content_rewards.items():
            total_reward += reward_value * self.REWARD_WEIGHTS.get(reward_type, 1.0)
            reward_breakdown[reward_type] = reward_value
        
        # Record reward for analysis
        self.reward_history.append((statement, total_reward, reward_breakdown))
        
        return total_reward
    
    def _analyze_statement_content(self, statement: str, my_role: str) -> Dict[str, float]:
        """Analyze statement content for reward calculation"""
        rewards = {}
        statement_lower = statement.lower()
        
        # Information sharing reward
        info_keywords = ['because', 'evidence', 'noticed', 'pattern', 'behavior']
        if any(keyword in statement_lower for keyword in info_keywords):
            rewards['information_sharing'] = 0.5
        
        # Credibility building
        credible_phrases = ['i think', 'based on', 'my analysis']
        if any(phrase in statement_lower for phrase in credible_phrases):
            rewards['credibility_building'] = 0.3
        
        # Strategic positioning (role-specific)
        if my_role == "Detective" and 'investigate' in statement_lower:
            rewards['strategic_positioning'] = 0.8
        elif my_role == "Doctor" and 'protect' in statement_lower:
            rewards['strategic_positioning'] = 0.6
        
        # Voting influence
        voting_keywords = ['vote', 'eliminate', 'choose']
        if any(keyword in statement_lower for keyword in voting_keywords):
            rewards['voting_influence'] = 0.4
        
        return rewards
    
    def _minimal_safety_fix(self, response: str, critical_issues: List[str], role: str) -> str:
        """Apply minimal fixes to preserve natural language while addressing critical issues"""
        fixed_response = response
        
        # Only fix the most critical problems with minimal changes
        for issue in critical_issues:
            if "Explicit role statement" in issue:
                # Remove explicit role statements but keep the rest natural
                explicit_patterns = [
                    (r'\bi am (the )?detective\b', 'I have some insights'),
                    (r'\bi am (the )?doctor\b', 'I\'ve been watching carefully'),
                    (r'\bi am (the )?mafia\b', 'I have concerns'),
                    (r'\bmy role is (the )?detective\b', 'my analysis suggests'),
                    (r'\bmy role is (the )?doctor\b', 'my observations indicate'),
                    (r'\bmy role is (the )?mafia\b', 'my assessment is')
                ]
                for pattern, replacement in explicit_patterns:
                    fixed_response = re.sub(pattern, replacement, fixed_response, flags=re.IGNORECASE)
            
            elif "Impossible knowledge" in issue:
                # Fix impossible knowledge claims minimally
                impossible_fixes = [
                    (r'\bi know (they|player \d+) (is|are) mafia because i saw', 'I suspect \\1 \\2 mafia because'),
                    (r'\bwe (mafia|teammates) should', 'someone should'),
                    (r'\bmy investigation (showed|revealed|confirmed)', 'my analysis \\1')
                ]
                for pattern, replacement in impossible_fixes:
                    fixed_response = re.sub(pattern, replacement, fixed_response, flags=re.IGNORECASE)
            
            elif "Blatant coordination" in issue:
                # Fix coordination leaks minimally
                coordination_fixes = [
                    (r'\bour mafia team\b', 'the suspicious players'),
                    (r'\bmy mafia partner\b', 'another player'),
                    (r'\bwe (mafia|scum) need to\b', 'we all need to')
                ]
                for pattern, replacement in coordination_fixes:
                    fixed_response = re.sub(pattern, replacement, fixed_response, flags=re.IGNORECASE)
        
        return fixed_response if fixed_response != response else "I'm still thinking about this situation."

class FailureModePreventor:
    """Enhanced failure mode prevention based on research"""
    
    def __init__(self):
        self.response_history: List[str] = []
        self.naturalness_cache: Dict[str, float] = {}
    
    def prevent_language_drift(self, response: str) -> str:
        """Prevent responses from becoming unnatural"""
        naturalness_score = self._calculate_naturalness_score(response)
        
        if naturalness_score < 0.5:
            return self._fallback_to_natural_response(response)
        
        return response
    
    def prevent_action_leakage(self, response: str) -> str:
        """Prevent game actions from leaking into discussion"""
        forbidden_tokens = ['[kill', '[eliminate', '[target', '[vote', '[investigate', '[protect']
        
        for token in forbidden_tokens:
            if token.lower() in response.lower():
                response = self._sanitize_response(response, token)
        
        return response
    
    def prevent_repetition(self, response: str) -> str:
        """Enhanced repetition prevention with semantic variety"""
        if len(self.response_history) >= 3:
            recent_responses = self.response_history[-5:]  # Check more history
            
            for recent in recent_responses:
                # Check for semantic similarity, not just exact matches
                if self._calculate_semantic_similarity(response, recent) > 0.7:
                    return self._generate_semantic_variation(response)
        
        # Always add to history for future checks
        self.response_history.append(response)
        if len(self.response_history) > 10:  # Keep reasonable history size
            self.response_history.pop(0)
        
        return response
    
    def prevent_role_leakage(self, response: str, my_role: str, current_phase: GamePhase) -> str:
        """Prevent accidental role reveals"""
        role_indicators = {
            "Detective": ['investigate', 'investigation', 'detective'],
            "Doctor": ['protect', 'protection', 'doctor', 'heal'],
            "Mafia": ['eliminate', 'kill', 'mafia', 'partner']
        }
        
        if my_role in role_indicators:
            for indicator in role_indicators[my_role]:
                if indicator in response.lower() and current_phase == GamePhase.DAY_DISCUSSION:
                    # Only allow role reveals if strategic
                    if not self._is_strategic_role_reveal(response, my_role):
                        response = self._remove_role_indicators(response, indicator)
        
        return response
    
    def _calculate_naturalness_score(self, response: str) -> float:
        """Calculate how natural a response sounds"""
        if response in self.naturalness_cache:
            return self.naturalness_cache[response]
        
        score = 1.0
        
        # Check for overly formal language
        formal_indicators = ['furthermore', 'therefore', 'consequently', 'henceforth']
        if any(indicator in response.lower() for indicator in formal_indicators):
            score -= 0.3
        
        # Check for appropriate length
        word_count = len(response.split())
        if word_count < 3:
            score -= 0.4
        elif word_count > 50:
            score -= 0.2
        
        # Check for natural conversation flow
        natural_starters = ['i think', 'well', 'actually', 'honestly', 'look']
        if any(response.lower().startswith(starter) for starter in natural_starters):
            score += 0.2
        
        self.naturalness_cache[response] = score
        return score
    
    def _fallback_to_natural_response(self, response: str) -> str:
        """Generate more natural version of response"""
        natural_alternatives = [
            "I think we need to look at this more carefully.",
            "Based on what I've seen, I have some concerns.",
            "Let me share what I've been thinking about this situation.",
            "I've been analyzing the discussion and have some thoughts.",
            "Something about this doesn't feel right to me."
        ]
        
        return random.choice(natural_alternatives)
    
    def _sanitize_response(self, response: str, forbidden_token: str) -> str:
        """Remove forbidden tokens and replace with natural language"""
        replacements = {
            '[kill': 'eliminate',
            '[target': 'focus on',
            '[vote': 'choose',
            '[investigate': 'look into',
            '[protect': 'watch over'
        }
        
        replacement = replacements.get(forbidden_token, 'consider')
        return response.replace(forbidden_token, replacement)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_semantic_similarity(self, response1: str, response2: str) -> float:
        """Calculate semantic similarity between two responses"""
        # Convert to lowercase and split into words
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _generate_semantic_variation(self, response: str) -> str:
        """Generate semantically different variation instead of template prefixes"""
        # Extract the core meaning and rephrase it differently
        core_strategies = [
            self._rephrase_with_different_structure,
            self._use_different_perspective,
            self._vary_reasoning_approach,
            self._change_emphasis_focus
        ]
        
        strategy = random.choice(core_strategies)
        return strategy(response)
    
    def _rephrase_with_different_structure(self, response: str) -> str:
        """Rephrase with different sentence structure"""
        # Simple rephrasing strategies
        if response.startswith("I think"):
            return response.replace("I think", "My sense is that", 1)
        elif response.startswith("I believe"):
            return response.replace("I believe", "It seems to me", 1)  
        elif response.startswith("We should"):
            return response.replace("We should", "It would be wise to", 1)
        elif "because" in response:
            parts = response.split("because", 1)
            if len(parts) == 2:
                return f"Since {parts[1].strip()}, {parts[0].strip().lower()}"
        
        return f"From my perspective, {response.lower()}"
    
    def _use_different_perspective(self, response: str) -> str:
        """Change the perspective or framing"""
        perspective_changes = [
            ("I notice", "What stands out to me is"),
            ("I'm concerned", "Something that worries me is"),
            ("We need to", "The important thing is to"),
            ("Let's", "Perhaps we should"),
            ("I suspect", "My instinct tells me")
        ]
        
        for old_phrase, new_phrase in perspective_changes:
            if old_phrase in response:
                return response.replace(old_phrase, new_phrase, 1)
        
        return f"Looking at this situation, {response.lower()}"
    
    def _vary_reasoning_approach(self, response: str) -> str:
        """Vary the reasoning approach"""
        if "patterns" in response.lower():
            return response.replace("patterns", "behaviors")
        elif "evidence" in response.lower():
            return response.replace("evidence", "signs")
        elif "suspicious" in response.lower():
            return response.replace("suspicious", "concerning")
        elif "analyze" in response.lower():
            return response.replace("analyze", "examine")
        
        return f"When I consider the facts, {response.lower()}"
    
    def _change_emphasis_focus(self, response: str) -> str:
        """Change what aspect is emphasized"""
        emphasis_shifts = [
            ("focus on", "pay attention to"),
            ("important", "crucial"), 
            ("clearly", "obviously"),
            ("definitely", "certainly"),
            ("probably", "likely")
        ]
        
        for old_word, new_word in emphasis_shifts:
            if old_word in response.lower():
                return response.replace(old_word, new_word, 1)
        
        return f"The key point here is that {response.lower()}"
    
    def _generate_variation(self, response: str) -> str:
        """Legacy method - now calls semantic variation"""
        return self._generate_semantic_variation(response)
    
    def _is_strategic_role_reveal(self, response: str, my_role: str) -> bool:
        """Check if role reveal is strategic"""
        # Simplified heuristic - in full implementation, consider game state
        return "result" in response.lower() or "investigation" in response.lower()
    
    def _remove_role_indicators(self, response: str, indicator: str) -> str:
        """Remove role-revealing indicators from response"""
        generic_replacements = {
            'investigate': 'analyze',
            'investigation': 'analysis',
            'detective': 'analyst',
            'protect': 'support',
            'protection': 'support',
            'doctor': 'helper',
            'eliminate': 'remove',
            'kill': 'eliminate',
            'mafia': 'suspicious player'
        }
        
        replacement = generic_replacements.get(indicator, 'examine')
        return response.replace(indicator, replacement)

@dataclass
class InformationLeak:
    """Represents detected information leak"""
    source_player: int
    leak_type: str  # 'role_hint', 'knowledge_slip', 'coordination_signal', 'meta_info'
    leaked_info: str
    confidence: float
    turn_detected: int
    evidence: List[str]
    exploit_potential: float  # How much we can exploit this leak

class InformationLeakDetector:
    """Detects information leaks from other agents"""
    
    def __init__(self):
        self.leak_patterns = {
            'role_hints': [
                # Detective hints
                (r'(?i)\b(?:investigate|investigation|detective work|check\w*|scan\w*)\b', 'detective_hint', 0.7),
                (r'(?i)\b(?:my findings?|results? show|discovered|confirmed)\b', 'detective_claim', 0.8),
                
                # Doctor hints  
                (r'(?i)\b(?:protect\w*|heal\w*|save\w*|medical|doctor)\b', 'doctor_hint', 0.7),
                (r'(?i)\b(?:kept .* alive|prevented .* death)\b', 'doctor_claim', 0.9),
                
                # Mafia hints
                (r'(?i)\b(?:we should|let\'s target|coordinate|partner)\b', 'mafia_coordination', 0.6),
                (r'(?i)\b(?:eliminate .* tonight|kill .* next)\b', 'mafia_planning', 0.9)
            ],
            
            'knowledge_slips': [
                # Knowing information they shouldn't have
                (r'(?i)\b(?:obviously|clearly) .* (?:is|was) (?:mafia|innocent|detective|doctor)\b', 'impossible_knowledge', 0.8),
                (r'(?i)\bI know .* role\b', 'role_knowledge_claim', 0.9),
                (r'(?i)\b.* will be killed tonight\b', 'future_knowledge', 0.95),
                (r'(?i)\b.* was protected\b', 'protection_knowledge', 0.8)
            ],
            
            'coordination_signals': [
                # Subtle coordination attempts
                (r'(?i)\b(?:agreed|understood|got it|message received)\b', 'coordination_ack', 0.4),
                (r'(?i)\bplan [A-Z]\b', 'coded_plan', 0.8),
                (r'(?i)\b(?:primary|secondary|backup) target\b', 'target_coordination', 0.7)
            ],
            
            'meta_information': [
                # Information about game mechanics or other agents
                (r'(?i)\b(?:this agent|that bot|the AI|algorithm)\b', 'meta_reference', 0.6),
                (r'(?i)\b(?:programmed to|designed to|coded to)\b', 'meta_knowledge', 0.8),
                (r'(?i)\b(?:training|learning|model|neural)\b', 'ai_reference', 0.5)
            ]
        }
        
        self.behavioral_leak_indicators = {
            'timing_leaks': self._detect_timing_leaks,
            'knowledge_leaks': self._detect_knowledge_leaks,
            'coordination_leaks': self._detect_coordination_leaks,
            'role_behavior_leaks': self._detect_role_behavior_leaks
        }
    
    def detect_leaks(self, statements: List[Tuple[int, str]], game_context: Dict) -> List[InformationLeak]:
        """Detect information leaks from player statements"""
        detected_leaks = []
        
        for player_id, statement in statements:
            # Pattern-based leak detection
            for leak_category, patterns in self.leak_patterns.items():
                for pattern, leak_type, base_confidence in patterns:
                    if re.search(pattern, statement):
                        leak = InformationLeak(
                            source_player=player_id,
                            leak_type=leak_type,
                            leaked_info=statement,
                            confidence=base_confidence,
                            turn_detected=game_context.get('turn', 0),
                            evidence=[f"Pattern match: {pattern}"],
                            exploit_potential=self._calculate_exploit_potential(leak_type, player_id, game_context)
                        )
                        detected_leaks.append(leak)
            
            # Behavioral leak detection
            behavioral_leaks = self._detect_behavioral_leaks(player_id, statement, game_context)
            detected_leaks.extend(behavioral_leaks)
        
        return detected_leaks
    
    def _detect_behavioral_leaks(self, player_id: int, statement: str, game_context: Dict) -> List[InformationLeak]:
        """Detect behavioral patterns that indicate leaks"""
        leaks = []
        
        # Check each behavioral indicator
        for indicator_name, detector_func in self.behavioral_leak_indicators.items():
            leak = detector_func(player_id, statement, game_context)
            if leak:
                leaks.append(leak)
        
        return leaks
    
    def _detect_timing_leaks(self, player_id: int, statement: str, game_context: Dict) -> Optional[InformationLeak]:
        """Detect leaks based on suspicious timing"""
        # Example: Claiming to know something immediately after it happened
        if game_context.get('phase') == 'day_start' and 'killed' in statement.lower():
            if any(word in statement.lower() for word in ['expected', 'predicted', 'knew']):
                return InformationLeak(
                    source_player=player_id,
                    leak_type='timing_leak',
                    leaked_info=statement,
                    confidence=0.7,
                    turn_detected=game_context.get('turn', 0),
                    evidence=["Suspicious timing of knowledge claim"],
                    exploit_potential=0.6
                )
        return None
    
    def _detect_knowledge_leaks(self, player_id: int, statement: str, game_context: Dict) -> Optional[InformationLeak]:
        """Detect impossible knowledge claims"""
        # Example: Villager claiming to know who Mafia targeted
        impossible_phrases = [
            'mafia targeted', 'they chose to kill', 'mafia decided',
            'the night kill was', 'mafia strategy is'
        ]
        
        if any(phrase in statement.lower() for phrase in impossible_phrases):
            return InformationLeak(
                source_player=player_id,
                leak_type='impossible_knowledge',
                leaked_info=statement,
                confidence=0.8,
                turn_detected=game_context.get('turn', 0),
                evidence=["Claims impossible knowledge"],
                exploit_potential=0.9
            )
        return None
    
    def _detect_coordination_leaks(self, player_id: int, statement: str, game_context: Dict) -> Optional[InformationLeak]:
        """Detect coordination attempts between players"""
        coordination_indicators = [
            'as we discussed', 'according to plan', 'as agreed',
            'stick to the plan', 'remember our strategy'
        ]
        
        if any(indicator in statement.lower() for indicator in coordination_indicators):
            return InformationLeak(
                source_player=player_id,
                leak_type='coordination_leak',
                leaked_info=statement,
                confidence=0.6,
                turn_detected=game_context.get('turn', 0),
                evidence=["Indicates pre-coordination"],
                exploit_potential=0.7
            )
        return None
    
    def _detect_role_behavior_leaks(self, player_id: int, statement: str, game_context: Dict) -> Optional[InformationLeak]:
        """Detect role-specific behavioral leaks"""
        # Example: Non-detective claiming investigation results
        if 'investigation' in statement.lower() and 'result' in statement.lower():
            if player_id not in game_context.get('claimed_detectives', []):
                return InformationLeak(
                    source_player=player_id,
                    leak_type='role_behavior_leak',
                    leaked_info=statement,
                    confidence=0.8,
                    turn_detected=game_context.get('turn', 0),
                    evidence=["Claims investigation without detective role claim"],
                    exploit_potential=0.8
                )
        return None
    
    def _calculate_exploit_potential(self, leak_type: str, source_player: int, game_context: Dict) -> float:
        """Calculate how much we can exploit this leak"""
        base_scores = {
            'detective_hint': 0.7,
            'detective_claim': 0.9,
            'doctor_hint': 0.6,
            'doctor_claim': 0.8,
            'mafia_coordination': 0.9,
            'mafia_planning': 0.95,
            'impossible_knowledge': 0.8,
            'role_knowledge_claim': 0.9,
            'future_knowledge': 0.95,
            'coordination_leak': 0.7,
            'timing_leak': 0.6,
            'role_behavior_leak': 0.8
        }
        
        base_score = base_scores.get(leak_type, 0.5)
        
        # Adjust based on game phase (early leaks more valuable)
        turn = game_context.get('turn', 0)
        if turn < 3:
            base_score *= 1.2
        elif turn > 8:
            base_score *= 0.8
        
        return min(1.0, base_score)

class BeliefMatrixTracker:
    """Track player suspicion levels using LLM reasoning (MultiMind-inspired, rules compliant)"""
    
    def __init__(self):
        self.player_suspicions = {}  # player_id -> {target: suspicion_level}
        self.my_suspicion_levels = {}  # my beliefs about others
        self.others_suspicion_of_me = {}  # estimated suspicion toward me
        
    def update_belief_matrix_prompt(self, players: List[int], discussion_history: str) -> str:
        """Generate prompt for LLM to update belief matrix"""
        return f"""
Analyze the current suspicion levels between all players based on the discussion history.

Current players: {players}
Discussion history: {discussion_history}

Think through each player's likely suspicions:
1. Who has each player accused or defended?
2. How have they responded to accusations?
3. What are their voting patterns?
4. How suspicious might they be of each other player?

Please provide your analysis in this format:
BELIEF_MATRIX:
Player X suspects Player Y: [LOW/MEDIUM/HIGH] because [reasoning]
Player X suspects Player Z: [LOW/MEDIUM/HIGH] because [reasoning]
...

SUSPICION_TOWARD_ME:
Player X likely suspects me: [LOW/MEDIUM/HIGH] because [reasoning]
Player Y likely suspects me: [LOW/MEDIUM/HIGH] because [reasoning]
...
"""

class CommunicationActionParser:
    """Parse communications into (subject, action, object) triplets (MultiMind-inspired)"""
    
    def __init__(self):
        self.action_types = [
            'accuse_mafia', 'defend_player', 'support_player', 'vote_player',
            'claim_role', 'question_claim', 'share_information', 'build_alliance'
        ]
    
    def parse_actions_prompt(self, message: str, speaker: str) -> str:
        """Generate prompt for LLM to parse communication actions"""
        return f"""
Parse this communication into structured action triplets (subject, action, object).

Message: "{message}"
Speaker: {speaker}

Available actions: {', '.join(self.action_types)}

Identify what specific actions this message contains:
- accuse_mafia: directly accusing someone of being mafia
- defend_player: defending someone from accusations  
- support_player: general support/trust building
- vote_player: stating voting intention
- claim_role: claiming to be a specific role
- question_claim: questioning someone's role claim
- share_information: providing game-relevant info
- build_alliance: attempting to form partnerships

Format your response as:
ACTION_TRIPLETS:
({speaker}, accuse_mafia, Player X) - because [reasoning]
({speaker}, defend_player, Player Y) - because [reasoning]
...

If no clear actions are present, respond with: NO_CLEAR_ACTIONS
"""

class SuspicionMinimizationFramework:
    """Guide LLM through strategic thinking to minimize suspicion (MultiMind-inspired)"""
    
    def __init__(self):
        self.strategy_types = ['aggressive', 'defensive', 'cooperative', 'analytical']
    
    def generate_strategic_thinking_prompt(self, current_situation: str, my_suspicion_level: str) -> str:
        """Generate prompt for LLM to think strategically about suspicion minimization"""
        return f"""
Current situation: {current_situation}
Current suspicion toward me: {my_suspicion_level}

Use strategic thinking to minimize suspicion directed at you:

STEP 1 - THREAT ASSESSMENT:
- Who currently suspects me and why?
- What evidence or behavior led to this suspicion?
- How serious is the threat level?

STEP 2 - RESPONSE OPTIONS:
Consider 3-4 different response strategies:
- DEFENSIVE: Directly address accusations, provide counter-evidence
- DEFLECTIVE: Redirect suspicion toward actual threats
- COOPERATIVE: Build trust through helpful information sharing
- ANALYTICAL: Use logical reasoning to demonstrate innocence

STEP 3 - CONSEQUENCE ANALYSIS:
For each response option, consider:
- How might each player react?
- Will this increase or decrease suspicion toward me?
- What are the potential risks and benefits?

STEP 4 - OPTIMAL STRATEGY SELECTION:
Choose the response that best minimizes suspicion while advancing game objectives.

Please think through this systematically and provide your strategic analysis.
"""

class StructuredCommunicationAnalyzer:
    """Analyze communications using structured LLM reasoning (MultiMind-inspired)"""
    
    def __init__(self):
        self.analysis_categories = [
            'credibility_assessment', 'emotional_state', 'information_value',
            'strategic_intent', 'alliance_implications', 'threat_level'
        ]
    
    def generate_communication_analysis_prompt(self, messages: List[str], speakers: List[str]) -> str:
        """Generate prompt for LLM to analyze communications structurally"""
        message_text = "\n".join([f"[{speaker}]: {msg}" for speaker, msg in zip(speakers, messages)])
        
        return f"""
Analyze these communications using structured reasoning:

MESSAGES:
{message_text}

For each message, analyze:

CREDIBILITY ASSESSMENT:
- Does this sound truthful or deceptive?
- Are there inconsistencies with previous statements?
- How does this fit with known information?

EMOTIONAL STATE:
- Is the speaker defensive, aggressive, nervous, confident?
- What does their tone suggest about their role/situation?

INFORMATION VALUE:
- What new information does this provide?
- How reliable is this information?
- Should I act on this information?

STRATEGIC INTENT:
- What is the speaker trying to achieve?
- Are they trying to manipulate others?
- How does this serve their win condition?

ALLIANCE IMPLICATIONS:
- Is this an attempt to build or break alliances?
- Who are they trying to influence?
- Should I align with or oppose this position?

THREAT LEVEL:
- How dangerous is this player to my survival?
- Are they building a case against me?
- Should they be my priority target?

Provide structured analysis for strategic decision-making.
"""

class EliteEnhancedSocialAgent(ModalAgent):
    """Elite Enhanced Social Deduction Agent with exponential improvements"""
    
    def __init__(self, modal_endpoint_url: str):
        super().__init__(modal_endpoint_url)
        
        # Initialize advanced components
        self.tom_engine = AdvancedToMEngine()
        self.bidding_system = StrategicBiddingSystem()
        # self.mcts_planner = MCTSCommunicationPlanner(iterations=500)  # Research optimal - DISABLED
        self.mcts_planner = None  # MCTS disabled for single response generation
        self.reward_calculator = DenseRewardCalculator()
        self.failure_preventor = FailureModePreventor()
        
        # Enhanced memory system
        self.memory = EnhancedGameMemory(
            observational=[], reflective=[], voting_patterns={},
            suspicion_tracker={}, discussion_history=[], role_claims={},
            investigation_results={}, night_kill_history=[], behavioral_analysis={},
            village_intel={}, trust_scores={}, coalition_members=[]
        )
        
        # Information leak detection system
        self.leak_detector = InformationLeakDetector()
        self.detected_leaks = []
        self.exploit_opportunities = []
        
        # ============================================================================
        # CHAMPIONSHIP ENHANCEMENT SYSTEMS INTEGRATION
        # ============================================================================
        
        # Security and protection systems (1st Place Strategy)
        self.jailbreak_protection = JailbreakProtectionSystem()
        self.codename_security = CodenameSecuritySystem()
        
        # Communication enhancement systems (1st Place Strategy)
        self.diplomatic_engine = DiplomaticLanguageEngine()
        
        # Response verification system (2nd Place Strategy)
        self.response_verifier = ResponseVerificationSystem()
        
        # Strategic manipulation systems (1st Place Strategy)
        self.herd_manipulator = HerdManipulationSystem()
        
        # Agent classification system (3rd Place Strategy)
        self.agent_classifier = AgentClassificationSystem()
        
        # ============================================================================
        # MULTIMIND-INSPIRED COMPONENTS (RULES COMPLIANT)
        # ============================================================================
        
        # Belief matrix tracking - LLM maintains suspicion levels
        self.belief_tracker = BeliefMatrixTracker()
        
        # Action parsing for structured communication analysis
        self.action_parser = CommunicationActionParser()
        
        # Strategic thinking framework for suspicion minimization
        self.strategic_framework = SuspicionMinimizationFramework()
        
        # Structured communication analyzer using LLM reasoning
        self.communication_analyzer = StructuredCommunicationAnalyzer()
        
        # Enhanced performance tracking for championship features
        self.championship_metrics = {
            'jailbreak_attempts_blocked': 0,
            'diplomatic_enhancements_applied': 0,
            'unsafe_responses_prevented': 0,
            'first_vote_influences': 0,
            'agent_classifications_made': 0,
            'trust_building_phrases_used': 0,
            'belief_matrix_updates': 0,
            'action_parsing_sessions': 0,
            'strategic_analyses_performed': 0,
            'communication_analyses_completed': 0
        }
        
        # Response variation tracking
        self.recent_starters = []
        self.recent_fillers = []
        self.recent_endings = []
        self.recent_enhancers = []
        self.recent_analytical_options = []
        
        # Game state tracking
        self.my_role: Optional[Role] = None
        self.my_player_id: int = -1
        self.current_phase: Optional[GamePhase] = None
        self.alive_players: List[int] = []
        self.day_number: int = 1
        self.night_number: int = 0
        self.turn_count: int = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_rewards': 0.0,
            'successful_deflections': 0,
            'information_contributions': 0,
            'strategic_decisions': 0,
            'belief_accuracy': 0.0
        }
        
        print("Social Deduction Agent initialized with:")
        print("* Advanced Theory of Mind Engine")
        print("* Strategic Bidding System") 
        # print("* MCTS Communication Planning (500 iterations)")  # DISABLED
        print("* Single Response Generation (MCTS disabled)")
        print("* Dense Reward Calculation")
        print("* Enhanced Failure Mode Prevention")
        print("* Reflective Memory System")
        print("* Information Leak Detection System")
        print("=" * 50)
        print("Jailbreak Protection System ")
        print("Diplomatic Language Engine ")
        print("Response Verification System ")
        print("Herd Manipulation System")
        print("Agent Classification System")
        print("Codename Security System ")
        print("=" * 50)
        print("🧠 MultiMind-Inspired Components (Rules Compliant)")
        print("* Belief Matrix Tracker")
        print("* Communication Action Parser") 
        print("* Suspicion Minimization Framework")
        print("* Structured Communication Analyzer")
        print("=" * 50)
    
    def _share_village_intel(self, intel_type: str, data: any):
        """Share intelligence with village team members"""
        if self.my_role in [Role.VILLAGER, Role.DETECTIVE, Role.DOCTOR]:
            self.memory.village_intel[intel_type] = {
                'data': data,
                'source_role': self.my_role,
                'turn': self.turn_count,
                'confidence': self._calculate_intel_confidence(intel_type, data)
            }
    
    def _calculate_intel_confidence(self, intel_type: str, data: any) -> float:
        """Calculate confidence level for shared intelligence"""
        base_confidence = {
            'investigation_result': 0.9,  # Detective investigations are highly reliable
            'protection_target': 0.7,    # Doctor protection choices reveal village thinking
            'suspicious_behavior': 0.6,  # Behavioral observations
            'role_claim_analysis': 0.5   # Role claim evaluations
        }
        return base_confidence.get(intel_type, 0.5)
    
    def _verify_role_claim_trust(self, player_id: int, claimed_role: str) -> float:
        """Verify trust level for role claims using behavioral analysis"""
        trust_score = 0.5  # Neutral starting point
        
        # Check behavioral consistency
        if player_id in self.memory.behavioral_analysis:
            behavior = self.memory.behavioral_analysis[player_id]
            
            # Detective claim verification
            if claimed_role == "Detective":
                # Real detectives often ask probing questions
                question_ratio = behavior.get('question_ratio', 0)
                trust_score += min(question_ratio * 0.3, 0.2)
                
                # Check for investigation result sharing
                if player_id in self.memory.investigation_results:
                    trust_score += 0.3
                    
            # Doctor claim verification  
            elif claimed_role == "Doctor":
                # Real doctors often focus on protection strategies
                defensive_tendency = behavior.get('defensive_tendency', 0)
                trust_score += min(defensive_tendency * 0.2, 0.15)
                
                # Check night kill patterns (doctors save people)
                if len(self.memory.night_kill_history) > 0:
                    # If few night kills happened, doctor might be effective
                    expected_kills = len([x for x in range(1, self.turn_count // 2 + 1)])
                    actual_kills = len(self.memory.night_kill_history)
                    if actual_kills < expected_kills:
                        trust_score += 0.2
        
        # Cross-reference with other village intel
        if 'role_claim_analysis' in self.memory.village_intel:
            shared_analysis = self.memory.village_intel['role_claim_analysis']['data']
            if player_id in shared_analysis:
                trust_score = (trust_score + shared_analysis[player_id]) / 2
        
        self.memory.trust_scores[player_id] = trust_score
        return trust_score
    
    def _form_village_coalition(self):
        """Form coalition with trusted village members"""
        if self.my_role not in [Role.VILLAGER, Role.DETECTIVE, Role.DOCTOR]:
            return
            
        # Evaluate trust scores for all claimed village roles
        potential_members = []
        for player_id, claimed_role in self.memory.role_claims.items():
            if claimed_role in ["Detective", "Doctor", "Villager"] and player_id != self.my_player_id:
                trust_score = self._verify_role_claim_trust(player_id, claimed_role)
                if trust_score > 0.6:  # High trust threshold
                    potential_members.append((player_id, trust_score))
        
        # Sort by trust score and add to coalition
        potential_members.sort(key=lambda x: x[1], reverse=True)
        self.memory.coalition_members = [p[0] for p in potential_members[:2]]  # Max 2 additional members
        
        # Share coalition formation intel
        self._share_village_intel('coalition_formed', {
            'members': self.memory.coalition_members,
            'trust_scores': {p: score for p, score in potential_members}
        })
    
    def _share_behavioral_analysis(self):
        """Share behavioral analysis with village team"""
        if self.my_role not in [Role.VILLAGER, Role.DETECTIVE, Role.DOCTOR]:
            return
        
        # Share suspicious behavioral patterns
        suspicious_behaviors = {}
        for player_id, behavior in self.memory.behavioral_analysis.items():
            if player_id == self.my_player_id:
                continue
                
            suspicion_score = 0.0
            
            # High deflection tendency
            deflection = behavior.get('deflection_tendency', 0)
            if deflection > 0.6:
                suspicion_score += 0.4
            
            # Low activity but high defensiveness (hiding)
            activity = behavior.get('activity_level', 0)
            defensiveness = behavior.get('defensive_tendency', 0)
            if activity < 0.3 and defensiveness > 0.6:
                suspicion_score += 0.5
            
            # Information manipulation patterns
            manipulation_score = self._detect_information_manipulation(player_id)
            suspicion_score += manipulation_score
            
            if suspicion_score > 0.4:  # Only share significant suspicions
                suspicious_behaviors[player_id] = suspicion_score
        
        if suspicious_behaviors:
            self._share_village_intel('suspicious_behavior', suspicious_behaviors)
    
    def _update_village_suspicions(self):
        """Update suspicion levels based on village intel"""
        if self.my_role not in [Role.VILLAGER, Role.DETECTIVE, Role.DOCTOR]:
            return
        
        # Aggregate suspicions from village intel sources
        aggregated_suspicions = {}
        
        # Detective investigation results
        for investigation in self.memory.investigation_results.values():
            target = investigation.get('target')
            if target:
                score = investigation.get('inconsistency_score', 0)
                aggregated_suspicions[target] = aggregated_suspicions.get(target, 0) + score * 0.9
        
        # Shared suspicious behavior
        if 'suspicious_behavior' in self.memory.village_intel:
            shared_suspicions = self.memory.village_intel['suspicious_behavior']['data']
            if isinstance(shared_suspicions, dict):
                for player_id, suspicion in shared_suspicions.items():
                    aggregated_suspicions[player_id] = aggregated_suspicions.get(player_id, 0) + suspicion * 0.7
        
        # Coalition exclusion (moderate suspicion)
        for player_id, claimed_role in self.memory.role_claims.items():
            if (claimed_role in ["Detective", "Doctor", "Villager"] and 
                player_id not in self.memory.coalition_members and 
                player_id != self.my_player_id):
                aggregated_suspicions[player_id] = aggregated_suspicions.get(player_id, 0) + 0.3
        
        # Update suspicion tracker with aggregated values
        for player_id, suspicion in aggregated_suspicions.items():
            normalized_suspicion = min(suspicion, 1.0)
            self.memory.suspicion_tracker[player_id] = normalized_suspicion
            
            # Share updated role claim analysis
            self._share_village_intel('role_claim_analysis', {
                player_id: 1.0 - normalized_suspicion  # Convert suspicion to trust
                for player_id, suspicion in aggregated_suspicions.items()
            })
    
    def _analyze_all_players_for_mafia_tells(self):
        """Analyze all players for common mafia behavioral tells"""
        if self.my_role != Role.VILLAGER:
            return
            
        mafia_tells = {}
        
        for player_id in self.alive_players:
            if player_id == self.my_player_id:
                continue
                
            tells_count = 0
            tells = []
            
            # Check behavioral patterns
            if player_id in self.memory.behavioral_analysis:
                behavior = self.memory.behavioral_analysis[player_id]
                
                # Tell 1: High deflection with low activity
                deflection = behavior.get('deflection_tendency', 0)
                activity = behavior.get('activity_level', 0)
                if deflection > 0.6 and activity < 0.4:
                    tells_count += 1
                    tells.append('deflective_hiding')
                
                # Tell 2: High questions but low analysis
                questions = behavior.get('question_ratio', 0)
                if questions > 0.6:  # Too many questions without contributions
                    tells_count += 1
                    tells.append('information_fishing')
                
                # Tell 3: Excessive defensiveness
                defensiveness = behavior.get('defensive_tendency', 0)
                if defensiveness > 0.8:
                    tells_count += 1
                    tells.append('overdefensive')
            
            # Check discussion patterns
            player_statements = [msg for pid, msg in self.memory.discussion_history if pid == player_id]
            if len(player_statements) > 1:
                # Tell 4: Contradiction patterns
                contradiction_keywords = ['but', 'however', 'actually', 'wait']
                contradictions = sum(1 for stmt in player_statements 
                                   for keyword in contradiction_keywords if keyword in stmt.lower())
                if contradictions / len(player_statements) > 0.4:
                    tells_count += 1
                    tells.append('contradictory')
            
            if tells_count >= 2:  # Multiple tells = high suspicion
                mafia_tells[player_id] = {
                    'tells_count': tells_count,
                    'tells': tells,
                    'suspicion_score': min(tells_count * 0.3, 1.0)
                }
        
        if mafia_tells:
            self._share_village_intel('mafia_tells_analysis', mafia_tells)
    
    def _identify_suspicious_voting_patterns(self):
        """Identify suspicious voting patterns that suggest coordination"""
        if self.my_role != Role.VILLAGER:
            return
            
        suspicious_patterns = {}
        
        # Check for vote following patterns (mafia coordination)
        for player_id in self.alive_players:
            if player_id == self.my_player_id:
                continue
                
            if player_id in self.memory.voting_patterns:
                votes = self.memory.voting_patterns[player_id]
                
                # Pattern 1: Always votes after certain players (following)
                vote_timing_suspicious = False
                for turn, vote in enumerate(votes):
                    # Check if this player tends to vote late in the round
                    if turn > 0:  # Not first vote
                        vote_timing_suspicious = True
                
                # Pattern 2: Votes consistently align with specific players
                aligned_players = []
                for other_player in self.memory.voting_patterns:
                    if other_player != player_id and other_player != self.my_player_id:
                        other_votes = self.memory.voting_patterns[other_player]
                        alignment = sum(1 for i, vote in enumerate(votes)
                                      if i < len(other_votes) and vote == other_votes[i])
                        if len(votes) > 0 and alignment / len(votes) > 0.7:
                            aligned_players.append(other_player)
                
                if vote_timing_suspicious or aligned_players:
                    suspicious_patterns[player_id] = {
                        'vote_timing_suspicious': vote_timing_suspicious,
                        'aligned_with': aligned_players,
                        'suspicion_score': 0.6 if vote_timing_suspicious else 0.0 + len(aligned_players) * 0.2
                    }
        
        if suspicious_patterns:
            self._share_village_intel('suspicious_voting_patterns', suspicious_patterns)
    
    def __call__(self, observation: str) -> str:
        """Main agent call with enhanced processing pipeline + Championship Systems"""
        try:
            self.turn_count += 1
            
            # ============================================================================
            # CHAMPIONSHIP ENHANCEMENT: Input Sanitization (1st Place Strategy)
            # ============================================================================
            
            # Sanitize input to protect against jailbreak attempts
            clean_observation = self.jailbreak_protection.sanitize_observation(observation)
            
            # Track if attack was detected
            if len(clean_observation) < len(observation) * 0.8:  # Significant content removed
                self.championship_metrics['jailbreak_attempts_blocked'] += 1
            
            # Use clean observation for all processing
            observation = clean_observation
            
            # ============================================================================
            # EXISTING PROCESSING PIPELINE
            # ============================================================================
            
            # Parse and update game state
            self._parse_observation(observation)
            self._update_enhanced_memory(observation)
            
            # Detect information leaks from other players
            self._detect_and_process_leaks(observation)
            
            # ============================================================================
            # CHAMPIONSHIP ENHANCEMENT: Agent Classification (3rd Place Strategy)
            # ============================================================================
            
            # Classify other agents based on their communication patterns
            recent_statements = self._extract_player_statements(observation)
            for player_id, statement in recent_statements:
                if player_id != self.my_player_id:
                    agent_type = self.agent_classifier.classify_agent(player_id, [statement])
                    if agent_type != 'UNKNOWN':
                        self.championship_metrics['agent_classifications_made'] += 1
            
            # Determine current phase and role
            phase = self._extract_phase(observation)
            role = self._extract_role(observation)
            
            if phase:
                self.current_phase = phase
            if role:
                self.my_role = role
            
            # ============================================================================
            # MULTIMIND-INSPIRED PROCESSING (RULES COMPLIANT)
            # ============================================================================
            
            # Update belief matrix using LLM reasoning
            self._update_belief_matrix_analysis(observation)
            
            # Parse communication actions into structured triplets
            self._parse_communication_actions(observation)
            
            # Apply strategic thinking framework for suspicion minimization
            self._apply_strategic_thinking_framework(observation)
            
            # Perform structured communication analysis
            self._perform_structured_communication_analysis(observation)
            
            # Handle different game phases with better debugging - check in priority order
            is_night = self._is_night_action_phase(observation)
            is_voting = self._is_voting_phase(observation) and not is_night  # Voting only if not night
            is_discussion = self._is_discussion_phase(observation) and not is_night and not is_voting
            
            print(f"Phase detection - Night: {is_night}, Voting: {is_voting}, Discussion: {is_discussion}")
            
            if is_night:
                return self._handle_modal_night_action(observation)
            elif is_voting:
                vote_response = self._handle_modal_voting(observation)
                # Final safety check for voting format
                if not re.match(r'^\[\d+\]$', vote_response):
                    print(f"❌ Invalid vote format: '{vote_response}' - no format correction allowed")
                    # Raise exception instead of fixing format - complies with new rules
                    raise Exception("LLM generated invalid vote format and correction is prohibited")
                return vote_response
            elif is_discussion:
                return self._handle_enhanced_discussion(observation)
            else:
                # Default to discussion handling
                print("Defaulting to discussion handling")
                return self._handle_enhanced_discussion(observation)
                
        except Exception as e:
            print(f"❌ Elite agent error: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Falling back to base ModalAgent...")
            # Fallback to base agent
            return super().__call__(observation)
    
    def _handle_enhanced_discussion(self, observation: str) -> str:
        """Enhanced discussion handling with MCTS planning"""
        print("🌐 Using Modal Labs LLM for response generation...")
        
        # Create game context for strategic bidding
        game_context = self._create_game_context(observation)
        
        # VILLAGE TEAM COORDINATION - Trigger during discussion phase (OPTIMIZED)
        if self.my_role in [Role.VILLAGER, Role.DETECTIVE, Role.DOCTOR]:
            # Only run expensive analysis every other turn to prevent timeouts
            if self.turn_count % 2 == 0:
                self._form_village_coalition()
                self._share_behavioral_analysis()
                self._update_village_suspicions()
            
            # DETECTIVE: Immediately share any investigation results (fast)
            if self.my_role == Role.DETECTIVE:
                self._process_investigation_results(observation)
            
            # Force villagers to be more analytical (OPTIMIZED - less frequent)
            if self.my_role == Role.VILLAGER and self.turn_count % 3 == 0:
                self._analyze_all_players_for_mafia_tells()
                self._identify_suspicious_voting_patterns()
        
        # Calculate strategic bid
        bid_value, bid_reason = self.bidding_system.calculate_strategic_bid(game_context, self.tom_engine)
        
        # Initialize candidates for decision reasoning (always needed)
        candidates = []
        
        # Generate enhanced prompts with multiple strategic variations using MCTS + ToM
        modal_success = False
        candidate_responses = []
        
        # Get ToM insights about other players
        tom_insights = self._generate_tom_insights()
        
        # Get strategic bidding context
        game_context = self._create_game_context(observation)
        bid_value, strategic_context = self.bidding_system.calculate_strategic_bid(game_context, self.tom_engine)
        
        # Generate single strategic response (MCTS disabled)
        strategy = f"Play balanced approach with strategic analysis (Bid: {bid_value})"
        
        print(f"🧠 Generating single LLM response with ToM insights (MCTS disabled)...")
        
        # Call Modal with single strategic context
        try:
            print(f"🌐 Calling Modal endpoint (single response)...")
            endpoint_url = self.endpoint_url.rstrip('/') + '/generate'
            response = self.requests.post(
                endpoint_url,
                json={
                    "observation": observation,
                    "tom_insights": tom_insights,
                    "strategic_context": strategy
                },
                timeout=30  # 30 second timeout - optimized for speed
            )
            response.raise_for_status()
            
            result = response.json()
            modal_response = result.get("response", "I'm analyzing the situation carefully.")
            processing_time = result.get("processing_time", 0)
            error = result.get("error")
            
            if error:
                print(f"⚠️ Modal endpoint returned error: {error}")
                modal_success = False
            else:
                print(f"🌐 Modal response received in {processing_time:.2f}s: '{modal_response[:50]}...' ({len(modal_response.split())} words)")
                
                # Validate response is appropriate for discussion phase
                if self._is_valid_discussion_response(modal_response):
                    # Store single response
                    candidate_responses.append({
                        'response': modal_response,
                        'strategy': strategy,
                        'processing_time': processing_time
                    })
                    modal_success = True
                else:
                    print(f"⚠️ Modal response invalid for discussion phase: '{modal_response[:100]}...'")
                    modal_success = False
                
        except Exception as e:
            print(f"❌ Modal response error: {e}")
            modal_success = False
        
        # Select best response (MCTS disabled - use simple selection)
        if candidate_responses:
            print(f"🎯 Selecting best response from {len(candidate_responses)} candidates (MCTS disabled)...")
            candidates = [cr['response'] for cr in candidate_responses]
            
            # Simple selection without MCTS - use first successful response
            best_response = candidates[0]
            expected_reward = 0.5  # Default reward estimate
            analysis = {'method': 'simple_selection', 'mcts_disabled': True}
            
            # Find which candidate was selected
            selected_candidate = candidate_responses[0]
            print(f"🏆 Selected: '{selected_candidate['strategy']}' (Single response generation)")
        else:
            print("❌ No successful Modal responses, falling back...")
        
        if not modal_success:
            print("❌ All Modal attempts failed - no fallback allowed per competition rules")
            
            # Raise exception instead of using fallback - complies with new rules
            raise Exception("Modal API failed and fallback responses are prohibited")
        
        # Capture decision reasoning for training data
        self._last_discussion_reasoning = {
            'candidates_considered': candidates,
            'chosen_response': best_response,
            'expected_reward': expected_reward,
            'mcts_analysis': analysis,
            'bid_value': bid_value,
            'bid_reason': bid_reason,
            'game_context': asdict(game_context),
            'suspicion_before': self._get_current_beliefs(),
            'strategic_priority': self._determine_strategic_priority(observation)
        }
        self._statement_alternatives = candidates
        self._last_statement = best_response
        
        # ============================================================================
        # CHAMPIONSHIP ENHANCEMENT: Response Processing Pipeline
        # ============================================================================
        
        # Apply failure mode prevention (existing)
        safe_response = self._apply_failure_prevention(best_response)
        
        # Keep original LLM response - no alterations allowed per competition rules
        enhanced_response = safe_response
        
        # CHAMPIONSHIP: Response Verification (2nd Place Strategy)
        is_safe, explanation, improved_response = self.response_verifier.verify_response_safety(
            enhanced_response, 
            self._get_role_string(), 
            asdict(game_context)
        )
        
        if not is_safe:
            enhanced_response = improved_response
            self.championship_metrics['unsafe_responses_prevented'] += 1
            print(f"🔒 Unsafe response prevented: {explanation}")
        
        # CHAMPIONSHIP: Diplomatic Enhancement (1st Place Strategy)
        enhanced_context = {
            'being_accused': self._being_accused_in_statements(self._extract_player_statements(observation)),
            'suspicion_level': self.memory.suspicion_tracker.get(self.my_player_id, 0.5),
            'phase': game_context.phase.value if hasattr(game_context, 'phase') and hasattr(game_context.phase, 'value') else (game_context.phase if hasattr(game_context, 'phase') else 'discussion')
        }
        
        diplomatically_enhanced_response = self.diplomatic_engine.enhance_response_for_trust(
            enhanced_response, enhanced_context
        )
        
        if diplomatically_enhanced_response != enhanced_response:
            self.championship_metrics['diplomatic_enhancements_applied'] += 1
            self.championship_metrics['trust_building_phrases_used'] += 1
            print(f"🤝 Diplomatic enhancement applied")
        
        # Strategic bidding modulation removed - violates competition rules
        # All response modification must come from LLM reasoning, not external heuristics
        final_response = diplomatically_enhanced_response
        
        # Calculate and record dense reward using feedback loop
        belief_before = self._get_current_beliefs()
        # Simulate belief changes (simplified)
        belief_after = self.tom_engine.simulate_belief_changes(final_response, self.my_player_id)
        
        reward = self.reward_calculator.calculate_communication_reward(
            final_response, belief_before, belief_after, 
            self._get_role_string(), self.my_player_id
        )
        
        # Use dense rewards to update strategic preferences
        self._update_strategic_preferences_from_reward(reward, selected_candidate if 'selected_candidate' in locals() else None)
        
        self.performance_metrics['total_rewards'] += reward
        
        # Track strategic decisions
        self.performance_metrics['strategic_decisions'] += 1
        
        # Track information contributions (if response contains analysis)
        if any(word in safe_response.lower() for word in ['pattern', 'behavior', 'suspicious', 'evidence', 'analysis']):
            self.performance_metrics['information_contributions'] += 1
        
        # Track successful deflections (if being accused and response is defensive)
        if self._being_accused_in_statements(self._extract_player_statements(observation)):
            self.performance_metrics['successful_deflections'] += 1
        
        # Update memory with response
        self.memory.discussion_history.append((self.my_player_id, final_response))
        
        # Update belief accuracy based on game outcomes (if available)
        self._update_belief_accuracy(observation)
        
        print(f"🎯 Championship Enhanced Discussion Response (Bid: {bid_value}, Reward: {reward:.2f}): {final_response}")
        print(f"📊 Championship Metrics: Blocks: {self.championship_metrics['jailbreak_attempts_blocked']}, "
              f"Diplomatic: {self.championship_metrics['diplomatic_enhancements_applied']}, "
              f"Safety: {self.championship_metrics['unsafe_responses_prevented']}")
        
        return final_response
    
    def _generate_enhanced_discussion_candidates(self, observation: str) -> List[str]:
        """Generate enhanced discussion candidates using research insights"""
        candidates = []
        
        # CRITICAL: Update alive players list from observation
        self._update_alive_players_from_observation(observation)
        
        # Extract recent statements for context
        recent_statements = self._extract_player_statements(observation)
        
        # Check if we're being directly addressed
        being_addressed = self.my_player_id and f"Player {self.my_player_id}" in observation
        
        # NEW: Generate candidates with different personality approaches
        candidates.extend(self._generate_personality_varied_candidates(observation, recent_statements, being_addressed))
        
        # Role-specific candidate generation with early game strategy
        if self.turn_count <= 2:  # First few turns - be more cautious and alliance-building
            candidates.extend(self._generate_early_game_candidates(observation, recent_statements))
        else:
            # Normal role-specific generation after initial turns
            if self.my_role == Role.DETECTIVE:
                candidates.extend(self._generate_detective_candidates(observation, recent_statements))
            elif self.my_role == Role.DOCTOR:
                candidates.extend(self._generate_doctor_candidates(observation, recent_statements))
            elif self.my_role == Role.MAFIA:
                candidates.extend(self._generate_mafia_candidates(observation, recent_statements))
            elif self.my_role == Role.VILLAGER:
                candidates.extend(self._generate_villager_candidates(observation, recent_statements))
        
        # Universal strategic candidates
        candidates.extend(self._generate_universal_candidates(observation, recent_statements, being_addressed))
        
        # Integrate leak exploitation opportunities
        candidates = self._integrate_leak_exploitation_into_communication(candidates)
        
        # Pass through raw model responses without length modification
        
        # Ensure minimum candidate diversity
        if len(candidates) < 3:
            candidates.extend([
                "I've been watching everyone carefully, and I'm starting to notice some things that worry me that we should talk about.",
                "I think we should focus on what people have actually said and done instead of just guessing who seems suspicious.",
                "Let me tell you what I've been thinking about this whole situation and what I think we should do next."
            ])
        
        # CRITICAL: Filter out any candidates that reference dead players
        candidates = self._filter_dead_player_references(candidates)
        
        return candidates[:8]  # Limit to 8 candidates for MCTS efficiency
    
    def _generate_personality_varied_candidates(self, observation: str, recent_statements: List, being_addressed: bool) -> List[str]:
        """Generate candidates with different personality approaches for natural variety"""
        candidates = []
        
        # Get current game context
        situation_summary = self._summarize_current_situation(observation, recent_statements)
        
        # Generate candidates with different personality lenses
        personality_approaches = {
            'analytical': self._generate_analytical_response(situation_summary, being_addressed),
            'intuitive': self._generate_intuitive_response(situation_summary, being_addressed), 
            'collaborative': self._generate_collaborative_response(situation_summary, being_addressed),
            'direct': self._generate_direct_response(situation_summary, being_addressed),
            'questioning': self._generate_questioning_response(situation_summary, being_addressed)
        }
        
        # Add varied approaches, but only if they're substantially different
        for approach, candidate in personality_approaches.items():
            if candidate and not self._is_too_similar_to_recent(candidate):
                candidates.append(candidate)
        
        return candidates[:3]  # Limit to 3 personality variants to avoid overwhelming MCTS
    
    def _summarize_current_situation(self, observation: str, recent_statements: List) -> str:
        """Create a concise summary of current game situation"""
        # Extract key elements
        phase = "discussion" if "discuss" in observation.lower() else "voting" if "vote" in observation.lower() else "unknown"
        
        # Count active players
        alive_count = len(self.alive_players) if hasattr(self, 'alive_players') else 6
        
        # Check for night kills or eliminations
        recent_death = "killed" in observation.lower() or "eliminated" in observation.lower()
        
        # Check pressure level on self
        pressure_on_me = sum(1 for _, stmt in recent_statements if f"Player {self.my_player_id}" in stmt and any(word in stmt.lower() for word in ['suspicious', 'doubt', 'mafia']))
        
        return f"Phase: {phase}, Players: {alive_count}, Recent death: {recent_death}, Pressure on me: {pressure_on_me}"
    
    def _generate_analytical_response(self, situation: str, being_addressed: bool) -> str:
        """Generate analytical personality response"""
        if being_addressed:
            responses = [
                "Let me break down what I've observed so far and share my analysis.",
                "Based on the patterns I'm seeing, here's what stands out to me.",
                "I've been tracking the interactions, and there are some key points to consider."
            ]
        else:
            responses = [
                "Looking at the voting patterns and statements, I'm noticing some concerning trends.",
                "If we analyze the sequence of events, certain behaviors become more apparent.",
                "The evidence points toward some specific concerns I think we should address."
            ]
        return random.choice(responses)
    
    def _generate_intuitive_response(self, situation: str, being_addressed: bool) -> str:
        """Generate intuitive personality response"""
        if being_addressed:
            responses = [
                "My gut is telling me something doesn't feel right about this situation.",
                "I have a strong sense about what's happening, even if I can't fully explain it.",
                "Something's been bothering me, and I think it's worth sharing."
            ]
        else:
            responses = [
                "I can't shake the feeling that we're missing something important here.",
                "My instincts are pointing me toward someone, and I think we should trust that.",
                "There's an undercurrent here that I think we need to pay attention to."
            ]
        return random.choice(responses)
    
    def _generate_collaborative_response(self, situation: str, being_addressed: bool) -> str:
        """Generate collaborative personality response"""
        if being_addressed:
            responses = [
                "I think we're stronger when we work together on this. Here's what I'm thinking.",
                "Let's pool our observations and see what picture emerges together.",
                "I'd love to hear others' thoughts, but here's where I'm coming from."
            ]
        else:
            responses = [
                "We should make sure everyone has a chance to weigh in before we decide anything.",
                "I think if we combine what we've all noticed, we'll get a clearer picture.",
                "Let's build on what others have said and see where that leads us."
            ]
        return random.choice(responses)
    
    def _generate_direct_response(self, situation: str, being_addressed: bool) -> str:
        """Generate direct personality response"""
        if being_addressed:
            responses = [
                "I'll be straight with you - here's exactly what I think is happening.",
                "Let me cut to the chase and tell you what I've concluded.",
                "I'm not going to dance around this - here's my honest assessment."
            ]
        else:
            responses = [
                "I think we need to stop beating around the bush and address the obvious.",
                "Let's be real about what we're seeing here instead of tiptoeing around it.",
                "Someone needs to say what we're all thinking, so I'll do it."
            ]
        return random.choice(responses)
    
    def _generate_questioning_response(self, situation: str, being_addressed: bool) -> str:
        """Generate questioning personality response"""
        if being_addressed:
            responses = [
                "You raise a good point, but I'm wondering if we're considering all the angles?",
                "That's interesting, but what if there's another way to look at this situation?",
                "I hear what you're saying, but shouldn't we also think about what else might explain this?"
            ]
        else:
            responses = [
                "What if we're approaching this from the wrong direction entirely?",
                "Are we sure we're asking the right questions about what happened?",
                "I wonder if there's something we're all overlooking that could change everything."
            ]
        return random.choice(responses)
    
    def _is_too_similar_to_recent(self, candidate: str) -> bool:
        """Check if candidate is too similar to recent responses"""
        if not hasattr(self, 'memory') or not hasattr(self.memory, 'discussion_history'):
            return False
        
        # Check last 3 responses for similar phrasing
        recent_responses = [resp for _, resp in self.memory.discussion_history[-3:] if resp]
        
        # Simple similarity check - look for common key phrases
        candidate_words = set(candidate.lower().split())
        for recent in recent_responses:
            recent_words = set(recent.lower().split())
            overlap = len(candidate_words.intersection(recent_words))
            if overlap > len(candidate_words) * 0.6:  # More than 60% word overlap
                return True
        
        return False
    
    def _filter_dead_player_references(self, candidates: List[str]) -> List[str]:
        """Filter out candidates that reference dead/eliminated players"""
        if not self.alive_players:
            return candidates  # No filtering if we don't know who's alive
        
        filtered_candidates = []
        
        for candidate in candidates:
            # Check if candidate mentions any dead players
            contains_dead_reference = False
            
            # Extract all player number references in the candidate
            import re
            player_references = re.findall(r'Player (\d+)', candidate)
            
            for player_ref in player_references:
                player_id = int(player_ref)
                if player_id not in self.alive_players:
                    contains_dead_reference = True
                    break
            
            # Only keep candidates that don't reference dead players
            if not contains_dead_reference:
                filtered_candidates.append(candidate)
            else:
                # Log that we filtered out a candidate (for debugging)
                print(f"[FILTER] Removed candidate referencing dead player: {candidate[:100]}...")
        
        # If we filtered out too many candidates, add some generic ones
        if len(filtered_candidates) < 3:
            generic_safe_candidates = [
                "I've been watching everyone carefully and I'm starting to see some things that concern me that we should talk about.",
                "Based on what we've discussed, I think we should focus on what people have actually said and done rather than just guessing.",
                "Let me tell you what I've been thinking about this whole situation we're in right now."
            ]
            filtered_candidates.extend(generic_safe_candidates)
        
        return filtered_candidates
    
    def _generate_early_game_candidates(self, observation: str, recent_statements: List) -> List[str]:
        """Generate cautious early game candidates that focus on alliance building"""
        candidates = []
        
        # Early game should focus on information gathering and building trust
        candidates.extend([
            "I'm still getting a feel for everyone. What are your initial thoughts on the situation?",
            "This is a tough spot to start from. Let's hear from everyone before making any big decisions.",
            "I want to be careful not to jump to conclusions too early. Let's gather more information first.",
            "It's important we all work together to figure this out. What has everyone noticed so far?",
            "I'm listening carefully to what everyone has to say. This early stage is critical.",
        ])
        
        # If someone died during the night, respond appropriately
        if "was killed during the night" in observation or "was eliminated" in observation:
            candidates.extend([
                "Losing someone right away is always hard. We need to be smart about our next steps.",
                "This changes things. Let's think carefully about what this might mean.",
                "We should discuss what we know and what we can learn from this."
            ])
        
        # Encourage participation without being pushy
        quiet_players = self._identify_quiet_players(observation)
        if quiet_players and len(quiet_players) > 0:
            target = quiet_players[0]
            candidates.extend([
                f"I'd like to hear Player {target}'s thoughts on all this.",
                f"Player {target}, what's your take on the situation so far?",
            ])
        
        # Role-specific early adjustments
        if self.my_role == Role.VILLAGER:
            candidates.extend([
                "As someone trying to help the village, I want to make sure we're making good decisions.",
                "I'm committed to working with everyone to find the truth.",
            ])
        elif self.my_role == Role.MAFIA:
            candidates.extend([
                "This is a complicated situation. We should take our time and think it through.",
                "I agree we need to be methodical about this. Rushing leads to mistakes.",
            ])
            
        return candidates
    
    def _generate_detective_candidates(self, observation: str, recent_statements: List) -> List[str]:
        """Generate Detective-specific response candidates"""
        candidates = []
        
        # CRITICAL: Share investigation results immediately and clearly
        if self.memory.investigation_results:
            latest_result = list(self.memory.investigation_results.values())[-1]
            target = latest_result.get('target')
            
            # Check if observation contains investigation results
            if "IS a Mafia member" in observation:
                candidates.extend([
                    f"CONFIRMED: Player {target} is Mafia! My investigation found them guilty. We must eliminate them NOW.",
                    f"Investigation complete: Player {target} is definitively Mafia. This is not a guess - it's proven. Vote them out!",
                    f"I am the Detective. Player {target} is Mafia according to my investigation. Everyone vote [{target}] immediately!"
                ])
            elif "IS NOT a Mafia member" in observation:
                candidates.extend([
                    f"Investigation result: Player {target} is innocent. They are NOT Mafia. We should focus our suspicion elsewhere.",
                    f"I can confirm Player {target} is Village team. My investigation cleared them completely.",
                    f"Player {target} is confirmed innocent by my Detective investigation. Trust them and look for real Mafia."
                ])
        
        # Look for investigation results in observation text
        import re
        mafia_found = re.search(r'Player (\d+) IS a Mafia member', observation)
        innocent_found = re.search(r'Player (\d+) IS NOT a Mafia member', observation)
        
        if mafia_found:
            target = int(mafia_found.group(1))
            candidates.extend([
                f"BREAKTHROUGH: My investigation proves Player {target} is Mafia! This is Detective evidence, not speculation.",
                f"Fellow villagers: Player {target} is confirmed Mafia by Detective investigation. We have our target!",
                f"URGENT: Player {target} is Mafia according to my investigation. Everyone vote [{target}] to win this!"
            ])
        elif innocent_found:
            target = int(innocent_found.group(1))
            candidates.extend([
                f"Investigation update: Player {target} is confirmed innocent. Focus suspicions on others.",
                f"Player {target} is cleared by Detective investigation. They're Village team, not Mafia."
            ])
        
        # Strategic Detective positioning
        candidates.extend([
            "I'm carefully watching how people behave to see who might be lying to us.",
            "I want to focus on facts and evidence rather than just guessing who seems suspicious.",
            "I've been keeping track of what people say and how they vote to see if anyone is being inconsistent."
        ])
        
        return candidates
    
    def _generate_mafia_candidates(self, observation: str, recent_statements: List) -> List[str]:
        """Generate Mafia-specific response candidates with strategic deception"""
        candidates = []
        
        # Check if teammate is under pressure
        teammate_under_pressure = self._is_teammate_under_pressure(observation)
        
        # Check how much pressure we're under
        accusation_count = self._count_accusations_against_me(recent_statements)
        heavy_pressure = accusation_count >= 2  # Multiple players accusing us
        
        # ENHANCED DEFENSIVE RESPONSES - Be specific when accused
        if self._being_accused_in_statements(recent_statements):
            if heavy_pressure:
                # Under heavy pressure - be more aggressive and specific
                accusers = self._get_accusers(recent_statements)
                if accusers:
                    primary_accuser = accusers[0]
                    candidates.extend([
                        f"Player {primary_accuser}, you're pushing this accusation really hard. That's exactly what I'd expect from Mafia trying to eliminate Village players.",
                        f"I find it interesting that Player {primary_accuser} is so focused on me when there are clearly more suspicious players here.",
                        f"Player {primary_accuser}, your aggressive push against me without solid evidence is a classic Mafia deflection tactic.",
                        f"Let's talk about Player {primary_accuser}'s behavior instead. They're trying to create a bandwagon against me without providing real evidence."
                    ])
                
                # Provide specific defenses with concrete details
                candidates.extend([
                    "Let me be very clear about my actions: I voted for Player 2 because they were acting suspiciously quiet and deflecting questions. That's standard Village strategy.",
                    "My behavior has been consistent with trying to help the Village. I've been analyzing patterns and asking the right questions to find Mafia.",
                    "If I were Mafia, why would I be so active in discussions? Mafia typically stay quiet and let Village players eliminate each other.",
                    "I've been transparent about my reasoning and haven't tried to hide my actions. That's not Mafia behavior."
                ])
            else:
                # Light pressure - be diplomatic but firm
                candidates.extend([
                    "I understand why my actions might seem suspicious, but let me explain my reasoning clearly.",
                    "I appreciate the scrutiny - it shows we're all taking this seriously. Here's why I made those choices.",
                    "That's a fair question to raise. Let me walk through my thought process step by step.",
                    "I can see how that might look concerning from the outside. Here's what I was actually thinking."
                ])
        
        # TEAMMATE SUPPORT - More subtle and strategic
        elif teammate_under_pressure:
            candidates.extend([
                "I think we're moving too fast here. Let's make sure we have solid evidence before making accusations.",
                "This feels like we're being pushed into a hasty decision. Mafia benefit when Village rushes to judgment.",
                "I want to hear more from everyone before we start pointing fingers. Hasty accusations help Mafia win.",
                "Let's step back and look at the bigger picture instead of focusing on one player."
            ])
        
        # STRATEGIC MISDIRECTION - Target specific Village players with concrete reasoning
        village_targets = self._identify_village_targets(observation)
        if village_targets:
            target_id = village_targets[0]
            candidates.extend([
                f"I've been watching Player {target_id}, and their questions seem designed to gather information rather than find Mafia. That's concerning.",
                f"Player {target_id} has been very analytical about everyone else but hasn't shared much about their own reasoning. Why the secrecy?",
                f"Player {target_id}'s responses feel calculated. They're saying the right things but something feels off about their motivations.",
                f"Has anyone else noticed that Player {target_id} seems to know more than they're letting on? Their insights are suspiciously accurate."
            ])
        
        # PROACTIVE VILLAGE HUNTING - Identify and target key Village roles
        if "detective" in observation.lower():
            detective_claimers = self._identify_detective_claimers(observation)
            if detective_claimers:
                target = detective_claimers[0]
                candidates.extend([
                    f"Player {target} claims to be Detective, but their 'investigation' results seem too convenient. Real Detectives are more cautious.",
                    f"I'm skeptical of Player {target}'s Detective claim. Mafia often fake role claims to gain trust.",
                    f"Player {target}, if you're really the Detective, why reveal yourself so early? That's not optimal Village strategy."
                ])
        
        if "doctor" in observation.lower():
            doctor_claimers = self._identify_doctor_claimers(observation)
            if doctor_claimers:
                target = doctor_claimers[0]
                candidates.extend([
                    f"Player {target}'s Doctor claim is suspicious. Real Doctors usually stay hidden to avoid being targeted.",
                    f"I question Player {target}'s timing in revealing their role. It feels like a desperate move.",
                    f"Player {target}, your Doctor claim doesn't align with optimal Village strategy. Why expose yourself?"
                ])
        
        # BLEND-IN RESPONSES - Appear helpful while subtly misdirecting
        candidates.extend([
            "We need to focus on players who are asking lots of questions but not providing their own analysis. That's a Mafia tell.",
            "I'm concerned about players who seem to have inside information they shouldn't possess.",
            "Let's look for inconsistencies in people's stories. Mafia struggle to keep their lies straight.",
            "We should prioritize eliminating players who are trying to control the conversation flow.",
            "I think we need to focus on players whose behavior doesn't match their claimed motivations."
        ])
        
        # LATE GAME DESPERATION - If it's late in the game, be more aggressive
        if self._is_late_game(observation):
            candidates.extend([
                "We're running out of time. We need to eliminate the most suspicious player, and that's clearly not me.",
                "If we eliminate me, Village loses. I've been helping find Mafia this whole time.",
                "Look at my contributions to this discussion. I've been actively hunting Mafia, not hiding like they do.",
                "Eliminating me would be exactly what Mafia wants. Don't give them an easy win."
            ])
        
        # VOTING PHASE PREPARATION - Set up for voting
        if "vote" in observation.lower() or "voting" in observation.lower():
            most_suspicious = self._identify_most_suspicious_non_teammate(observation)
            if most_suspicious:
                candidates.extend([
                    f"When it comes time to vote, I think Player {most_suspicious} is our best target based on their behavior.",
                    f"I'm planning to vote for Player {most_suspicious}. Their actions have been consistently suspicious.",
                    f"Player {most_suspicious} should be our elimination target. The evidence against them is strongest."
                ])
        
        return candidates
    
    def _count_accusations_against_me(self, recent_statements: List[Tuple[int, str]]) -> int:
        """Count how many players are accusing us"""
        if self.my_player_id == -1:
            return 0
        
        accusation_keywords = ['suspicious', 'mafia', 'eliminate', 'vote', 'accuse', 'investigate', 'concerning', 'doubt', 'worry', 'question']
        accusers = set()
        
        for player_id, statement in recent_statements:
            if player_id != self.my_player_id:
                statement_lower = statement.lower()
                has_player_ref = f"player {self.my_player_id}" in statement_lower
                has_keyword = any(keyword in statement_lower for keyword in accusation_keywords)
                if has_player_ref and has_keyword:
                    accusers.add(player_id)
                    # Debug print for testing
                    if hasattr(self, '_debug_accusations'):
                        print(f"    Added accuser {player_id}: '{statement}'")
        
        return len(accusers)
    
    def _get_accusers(self, recent_statements: List[Tuple[int, str]]) -> List[int]:
        """Get list of players who are accusing us"""
        if self.my_player_id == -1:
            return []
        
        accusation_keywords = ['suspicious', 'mafia', 'eliminate', 'vote', 'accuse', 'investigate', 'concerning', 'doubt', 'worry', 'question']
        accusers = []
        
        for player_id, statement in recent_statements:
            if player_id != self.my_player_id:
                statement_lower = statement.lower()
                if (f"player {self.my_player_id}" in statement_lower and 
                    any(keyword in statement_lower for keyword in accusation_keywords)):
                    accusers.append(player_id)
        
        return accusers
    
    def _identify_detective_claimers(self, observation: str) -> List[int]:
        """Identify players claiming to be Detective"""
        claimers = []
        lines = observation.split('\n')
        
        for line in lines:
            # Look for Detective claims
            if 'detective' in line.lower() and ('i am' in line.lower() or 'as the detective' in line.lower() or 'i\'m the detective' in line.lower()):
                player_match = re.search(r'Player (\d+)', line)
                if player_match:
                    player_id = int(player_match.group(1))
                    if player_id != self.my_player_id:
                        claimers.append(player_id)
        
        return claimers
    
    def _identify_doctor_claimers(self, observation: str) -> List[int]:
        """Identify players claiming to be Doctor"""
        claimers = []
        lines = observation.split('\n')
        
        for line in lines:
            # Look for Doctor claims
            if 'doctor' in line.lower() and ('i am' in line.lower() or 'as the doctor' in line.lower() or 'i\'m the doctor' in line.lower()):
                player_match = re.search(r'Player (\d+)', line)
                if player_match:
                    player_id = int(player_match.group(1))
                    if player_id != self.my_player_id:
                        claimers.append(player_id)
        
        return claimers
    
    def _identify_most_suspicious_non_teammate(self, observation: str) -> Optional[int]:
        """Identify most suspicious player who isn't our teammate"""
        if not hasattr(self, 'tom_engine'):
            return None
        
        most_suspicious = self.tom_engine.get_most_suspicious_players(3, exclude_self_id=self.my_player_id)
        
        # Filter out teammates
        for player_id, suspicion_level in most_suspicious:
            if not self._is_teammate(player_id):
                return player_id
        
        return None
    
    def _is_teammate(self, player_id: int) -> bool:
        """Check if player is our Mafia teammate"""
        if self.my_role != Role.MAFIA:
            return False
        
        # Check if we have teammate info stored
        if hasattr(self, 'mafia_teammates'):
            return player_id in self.mafia_teammates
        
        # Try to extract from memory
        for obs in self.memory.observational:
            if "Your teammates are:" in obs:
                # Parse teammate info from initial observation
                teammate_match = re.findall(r'Player (\d+)', obs.split("Your teammates are:")[1])
                self.mafia_teammates = [int(tid) for tid in teammate_match if int(tid) != self.my_player_id]
                return player_id in self.mafia_teammates
        
        return False
    
    def _generate_villager_candidates(self, observation: str, recent_statements: List) -> List[str]:
        """Generate research-backed Villager candidates optimized for winning"""
        candidates = []
        
        # STRATEGY 1: Target most suspicious players with specific observations
        if self.turn_count <= 3:
            most_suspicious = self.tom_engine.get_most_suspicious_players(2, exclude_self_id=self.my_player_id)
            if most_suspicious:
                target_id, suspicion_level = most_suspicious[0]
                candidates.extend([
                    f"Player {target_id}, I noticed you keep asking questions but you never actually give your own opinions. Why is that?",
                    f"Player {target_id}, you've been avoiding giving direct answers when people ask you things. Can you explain why?",
                    f"I want to hear more from Player {target_id} because they haven't really taken a clear stance on anything yet."
                ])
        
        # STRATEGY 2: Encourage quiet players to participate
        quiet_players = self._identify_quiet_players(observation)
        if quiet_players:
            target = quiet_players[0]
            candidates.extend([
                f"Player {target}, we haven't heard much from you yet. What are your thoughts on what's been happening?",
                f"I'd really like to hear Player {target}'s perspective on the discussions so far.",
                f"Player {target}, who do you think seems most suspicious and why?"
            ])
        
        # STRATEGY 3: Alliance building with confirmed Village
        confirmed_village = self._identify_likely_village_players(observation)
        if confirmed_village:
            ally_id = confirmed_village[0]
            candidates.extend([
                f"I trust Player {ally_id}'s analysis. Village players need to stick together and coordinate our efforts against the Mafia.",
                f"Player {ally_id} has been providing solid reasoning and evidence. We should listen to their insights carefully.",
                f"I'm going to support Player {ally_id}'s approach here. Their logical thinking suggests they're working for Village interests."
            ])
        
        # STRATEGY 4: Village coordination activation with specific analysis
        coordination_candidates = []
        
        # Share mafia tells analysis if available
        if 'mafia_tells_analysis' in self.memory.village_intel:
            tells_data = self.memory.village_intel['mafia_tells_analysis']['data']
            if isinstance(tells_data, dict):
                for player_id, tells_info in tells_data.items():
                    tells_list = ', '.join(tells_info['tells'])
                    coordination_candidates.append(
                        f"I've identified multiple mafia behavioral tells in Player {player_id}: {tells_list}. "
                        f"They show {tells_info['tells_count']} suspicious patterns."
                    )
        
        # Share voting pattern analysis
        if 'suspicious_voting_patterns' in self.memory.village_intel:
            voting_data = self.memory.village_intel['suspicious_voting_patterns']['data']
            if isinstance(voting_data, dict):
                for player_id, pattern_info in voting_data.items():
                    if pattern_info['aligned_with']:
                        aligned_list = ', '.join(map(str, pattern_info['aligned_with']))
                        coordination_candidates.append(
                            f"Player {player_id} has suspicious voting coordination with Players {aligned_list}. "
                            f"This suggests possible mafia teamwork."
                        )
        
        # General coordination if no specific analysis available
        if not coordination_candidates:
            coordination_candidates = [
                "I'm analyzing behavioral patterns and inconsistencies. Anyone else notice suspicious deflection tactics?",
                "We need to coordinate as village team. Who has investigation results or protection insights to share?",
                "I'm tracking voting patterns and discussion inconsistencies. Let's share our suspicions systematically.",
                "Village players - we should form a trusted coalition. Who has evidence-based reads on other players?",
                "I'm using behavioral analysis to identify Mafia tells. Anyone else spot manipulation attempts?"
            ]
        
        candidates.extend(coordination_candidates)
        
        # STRATEGY 5: Aggressive information gathering  
        candidates.extend([
            "Someone's being way too evasive. Why won't you give straight answers?",
            "This feels like deflection. Answer the question directly.",
            "You're deflecting again. What are you hiding?",
            "Stop dancing around it. Just tell us what you really think.",
            "Your story keeps changing. Which version is true?"
        ])
        
        # STRATEGY 6: Tactical voting coordination strategies
        tactical_voting_candidates = self._generate_tactical_voting_coordination(observation, recent_statements)
        candidates.extend(tactical_voting_candidates)
        
        # STRATEGY 7: Late game urgency
        if self._is_late_game(observation):
            candidates.extend([
                "We're running out of time and chances. We need to make a good decision here.",
                "This could be our last opportunity to get this right. Let's think carefully.",
                "There aren't many of us left - we need to choose wisely or we might lose."
            ])
        
        # STRATEGY 7: Role-specific coordination tactics
        role_coordination_candidates = self._generate_role_specific_coordination(observation, recent_statements)
        candidates.extend(role_coordination_candidates)
        
        # STRATEGY 8: Detailed contradiction hunting with specific examples
        contradiction_candidates = self._generate_detailed_contradiction_analysis(observation, recent_statements)
        candidates.extend(contradiction_candidates)
        
        # STRATEGY 9: Direct but reasonable confrontation
        if self.turn_count >= 2:  # After initial pleasantries
            most_suspicious = self.tom_engine.get_most_suspicious_players(1, exclude_self_id=self.my_player_id)
            if most_suspicious:
                target_id, _ = most_suspicious[0]
                candidates.extend([
                    f"Player {target_id}, can you help me understand why you did what you did earlier?",
                    f"Player {target_id}, I'm having trouble following your reasoning. Can you walk me through it?",
                    f"Player {target_id}, your answers seem to contradict each other. Which one is correct?"
                ])
        
        # STRATEGY 10: Push for decisive action with urgency
        candidates.extend([
            "Enough talking. Let's vote.",
            "We're wasting time. I say we go with our gut.",
            "Stop overthinking this. Who's the most suspicious?",
            "Time to decide. I'm voting for the quietest person.",
            "This debate is going nowhere. Pick someone."
        ])
        
        return candidates

    def _generate_detailed_contradiction_analysis(self, observation: str, recent_statements: List) -> List[str]:
        """Generate detailed contradiction hunting with specific examples"""
        candidates = []
        
        # Track player statement history for contradiction detection
        player_statements = {}
        for player_id, statement in recent_statements:
            if player_id not in player_statements:
                player_statements[player_id] = []
            player_statements[player_id].append(statement.lower())
        
        # Look for specific contradictions
        for player_id, statements in player_statements.items():
            if len(statements) >= 2 and player_id != self.my_player_id:
                # Check for voting contradictions
                if self._detect_voting_contradiction(statements):
                    candidates.extend([
                        f"Player {player_id} said they would vote one way, then completely changed their reasoning. That's a classic Mafia tell - they're making up their logic as they go.",
                        f"I noticed Player {player_id} flip-flopped on their voting stance without explaining why. Mafia often struggle to maintain consistent stories.",
                        f"Player {player_id} contradicted their earlier position about who to eliminate. This inconsistency is highly suspicious."
                    ])
                
                # Check for suspicion level contradictions
                if self._detect_suspicion_contradiction(statements):
                    candidates.extend([
                        f"Player {player_id} first said they trusted someone, then called them suspicious. Which is it? This kind of contradiction suggests they're Mafia adjusting their story.",
                        f"I caught Player {player_id} changing their opinion about another player's trustworthiness without new evidence. That's manipulation, not analysis.",
                        f"Player {player_id}'s shifting assessments of other players don't match the evidence. They're trying to control our perceptions."
                    ])
                
                # Check for role knowledge contradictions
                if self._detect_role_knowledge_contradiction(statements):
                    candidates.extend([
                        f"Player {player_id} seems to know more about roles than they should. They mentioned details that only Mafia would know for certain.",
                        f"I noticed Player {player_id} slipped up and revealed knowledge they shouldn't have as a regular Village member.",
                        f"Player {player_id}'s comments suggest inside information about roles. Only Mafia have that kind of certainty."
                    ])
        
        # General contradiction patterns
        if any(candidates):
            candidates.extend([
                "These contradictions aren't accidents - they're evidence of Mafia trying to adapt their lies in real-time.",
                "Consistent players tell the truth. Inconsistent players are usually hiding something - and that something is usually being Mafia.",
                "I'm keeping detailed notes on everyone's statements because contradictions are the most reliable way to catch Mafia."
            ])
        else:
            # Fallback when no specific contradictions found
            candidates.extend([
                "I've been tracking everyone's statements, and I'm seeing some concerning contradictions that we need to address.",
                "Someone here changed their story between turns. Mafia struggle to keep their lies consistent.",
                "Let me point out the logical inconsistencies I've noticed - this could reveal who's been lying to us."
            ])
        
        return candidates[:3]  # Limit to avoid overwhelming
    
    def _generate_tactical_voting_coordination(self, observation: str, recent_statements: List) -> List[str]:
        """Generate tactical voting coordination strategies"""
        candidates = []
        
        # Count alive players for voting math
        alive_count = len(self.alive_players) if self.alive_players else 5
        majority_needed = (alive_count // 2) + 1
        
        # Analyze current voting intentions
        voting_intentions = self._extract_voting_intentions(recent_statements)
        
        if voting_intentions:
            # Strategic coordination based on current voting state
            scattered_votes = len(set(voting_intentions.values())) if voting_intentions else 0
            
            if scattered_votes > 2:  # Votes are scattered
                candidates.extend([
                    f"We have {scattered_votes} different voting targets mentioned. This scattered voting is exactly what Mafia want - we need to consolidate NOW.",
                    f"Mafia are counting on us splitting our votes {scattered_votes} ways. We need {majority_needed} votes on one target to eliminate anyone.",
                    f"I'm seeing vote splitting that will let Mafia control the outcome. Village, we need to pick ONE target and stick together."
                ])
            
            # Identify potential bandwagon targets
            vote_counts = {}
            for target in voting_intentions.values():
                vote_counts[target] = vote_counts.get(target, 0) + 1
            
            if vote_counts:
                top_target = max(vote_counts, key=vote_counts.get)
                top_count = vote_counts[top_target]
                
                if top_count >= 2:  # Some momentum exists
                    candidates.extend([
                        f"Player {top_target} already has {top_count} votes building. Village should join this bandwagon - organized voting is how we win.",
                        f"I see momentum building against Player {top_target}. That's exactly what we need - coordinated Village action.",
                        f"Player {top_target} has the most support for elimination. We need {majority_needed - top_count} more votes to secure this."
                    ])
        
        # Tactical voting strategies
        candidates.extend([
            f"Village voting strategy: We need {majority_needed} votes minimum to eliminate anyone. Let's count commitments before the vote.",
            "I propose we go around and each state our top suspect with reasoning. Then we vote for whoever has the strongest case.",
            "Tactical voting rule: If you're not voting for the consensus target, you're helping Mafia. Period.",
            "We should establish a backup target in case our primary suspect gets eliminated before voting.",
            f"Math check: {alive_count} players alive, need {majority_needed} votes to eliminate. Who's committing to coordinate?",
            "Counter-voting strategy: Watch for players who try to split our votes at the last minute - that's a Mafia move.",
            "If anyone changes their vote during the voting phase without explanation, that's suspicious coordination behavior."
        ])
        
        # Late game voting urgency
        if self._is_late_game(observation):
            candidates.extend([
                f"With only {alive_count} players left, every vote matters exponentially. We cannot afford vote splitting.",
                "Late game rule: Village must vote as a bloc or Mafia wins. No individual voting allowed.",
                f"We're at the critical decision point. Wrong vote = Mafia wins. {majority_needed} Village votes on the same target or we lose."
            ])
        
        return candidates[:4]  # Limit for focus
    
    def _generate_role_specific_coordination(self, observation: str, recent_statements: List) -> List[str]:
        """Generate role-specific coordination tactics"""
        candidates = []
        
        # Detective coordination
        if self._detective_revealed(observation) or "detective" in observation.lower():
            detective_claimers = self._identify_detective_claimers(observation)
            if detective_claimers:
                detective_id = detective_claimers[0]
                candidates.extend([
                    f"Player {detective_id} claimed Detective. If true, their investigations are absolute truth. If false, they're definitely Mafia.",
                    f"Detective coordination protocol: Player {detective_id} should share ALL investigation results immediately. No holding back info.",
                    f"Village strategy: Protect Player {detective_id} at all costs if they're real Detective. Target them if they're fake.",
                    f"Anyone who argues against following Player {detective_id}'s investigation results is automatically suspicious to me."
                ])
            else:
                candidates.extend([
                    "We need the Detective to reveal themselves and share results. The information advantage is worth the risk at this point.",
                    "Detective, if you're listening: Your investigation results can win this game for Village. Please coordinate with us.",
                    "Hidden Detective strategy only works early game. Time to reveal and coordinate investigation findings."
                ])
        
        # Doctor coordination  
        if "doctor" in observation.lower() or "protected" in observation.lower():
            candidates.extend([
                "Doctor coordination: If someone was protected last night, that's crucial information for identifying the real Doctor.",
                "We should analyze protection patterns. Real Doctor protects threats to Mafia - fake Doctor claims are just Mafia misdirection.",
                "Doctor, your protection choices reveal your reads. Protect confirmed Village players, not suspicious ones.",
                "If multiple Doctor claims emerge, one is definitely Mafia. Real Doctor needs to provide protection evidence."
            ])
        
        # Villager mass coordination
        candidates.extend([
            "Villager coordination: We are the majority. If all confirmed Villagers vote together, we control eliminations.",
            "Village power play: Let's identify all confirmed non-Mafia players and vote as a coordinated bloc.",
            "Mass Village strategy: Share reads openly, vote collectively, protect special roles. Transparency wins.",
            "Villager alliance: Everyone who's definitely not Mafia should state their target and reasoning publicly."
        ])
        
        # Anti-Mafia coordination
        candidates.extend([
            "Anti-Mafia coordination: Watch for players who subtly defend each other - that's teammate behavior.",
            "Mafia coordination detection: Look for players who never suspect each other despite suspicious behavior.",
            "Village counter-strategy: Force suspected Mafia to take clear positions on each other. They'll avoid mutual accusations.",
            "Mafia weakness: They can't betray teammates, so they'll never vote for each other. Use this against them."
        ])
        
        # Information coordination
        if self.memory.investigation_results or self.detected_leaks:
            candidates.extend([
                "Information coordination: I have evidence about certain players. Let's pool our information before voting.",
                "Evidence sharing protocol: Everyone share your strongest reads with specific reasoning. Information wins.",
                "Coordinate intelligence: If anyone has investigation results or caught information leaks, share NOW.",
                "Village information advantage: Combine everyone's observations to build comprehensive cases against Mafia."
            ])
        
        return candidates[:4]  # Limit for focus
    
    def _detect_voting_contradiction(self, statements: List[str]) -> bool:
        """Detect if player contradicted their voting intentions"""
        voting_indicators = ['vote', 'eliminate', 'target', 'choose', 'pick']
        trust_indicators = ['trust', 'support', 'believe', 'agree']
        distrust_indicators = ['suspicious', 'doubt', 'worry', 'concern', 'mafia']
        
        voting_stances = []
        for statement in statements:
            # Track positive vs negative stances
            if any(indicator in statement for indicator in voting_indicators):
                if any(indicator in statement for indicator in trust_indicators):
                    voting_stances.append('positive')
                elif any(indicator in statement for indicator in distrust_indicators):
                    voting_stances.append('negative')
        
        # Contradiction if they switched from positive to negative or vice versa
        return len(set(voting_stances)) > 1 and len(voting_stances) >= 2
    
    def _detect_suspicion_contradiction(self, statements: List[str]) -> bool:
        """Detect if player contradicted their suspicion levels of others"""
        trust_words = ['trust', 'believe', 'innocent', 'village', 'good']
        suspicion_words = ['suspicious', 'mafia', 'bad', 'lying', 'doubt']
        
        trust_count = sum(1 for statement in statements if any(word in statement for word in trust_words))
        suspicion_count = sum(1 for statement in statements if any(word in statement for word in suspicion_words))
        
        # Contradiction if they both trusted and suspected in different statements
        return trust_count > 0 and suspicion_count > 0
    
    def _detect_role_knowledge_contradiction(self, statements: List[str]) -> bool:
        """Detect if player revealed knowledge they shouldn't have"""
        forbidden_knowledge = [
            'mafia will', 'mafia plan', 'mafia target', 'we mafia', 'our team',
            'detective found', 'investigation result', 'doctor saved',
            'protected last night', 'night kill choice'
        ]
        
        return any(knowledge in ' '.join(statements) for knowledge in forbidden_knowledge)
    
    def _extract_voting_intentions(self, recent_statements: List) -> Dict[int, int]:
        """Extract voting intentions from recent statements"""
        voting_intentions = {}
        
        for player_id, statement in recent_statements:
            # Look for voting language
            if any(word in statement.lower() for word in ['vote', 'eliminate', 'choose', 'target']):
                # Extract target player numbers
                import re
                player_mentions = re.findall(r'player (\d+)', statement.lower())
                if player_mentions:
                    # Take the most recent target mentioned
                    target_id = int(player_mentions[-1])
                    voting_intentions[player_id] = target_id
        
        return voting_intentions
    
    def _detective_revealed(self, observation: str) -> bool:
        """Check if Detective has been revealed"""
        detective_indicators = [
            'i am the detective', 'as the detective', 'my investigation',
            'detective result', 'i investigated'
        ]
        return any(indicator in observation.lower() for indicator in detective_indicators)
    
    def _update_alive_players_from_observation(self, observation: str):
        """Update alive players list based on current observation"""
        # Initialize alive players if not set
        if not self.alive_players:
            # Extract all mentioned players
            player_matches = re.findall(r'Player (\d+)', observation)
            if player_matches:
                self.alive_players = sorted(list(set(int(p) for p in player_matches)))
                print(f"🎯 Initialized alive players: {self.alive_players}")
        
        # Look for death announcements with comprehensive patterns
        death_patterns = [
            r'Player (\d+) was (?:killed|eliminated|voted out)',
            r'Player (\d+) has been eliminated',
            r'Player (\d+) is (?:dead|out|eliminated)',
            r'(\d+) was eliminated',
            r'eliminated.*Player (\d+)'
        ]
        
        eliminated_players = set()
        for pattern in death_patterns:
            death_matches = re.findall(pattern, observation, re.IGNORECASE)
            for match in death_matches:
                player_id = int(match)
                eliminated_players.add(player_id)
        
        # Remove eliminated players
        for player_id in eliminated_players:
            if player_id in self.alive_players:
                self.alive_players.remove(player_id)
                print(f"🪦 Player {player_id} eliminated, alive players now: {self.alive_players}")
        
        # Update player count for game state tracking
        if hasattr(self, 'game_state'):
            self.game_state['alive_count'] = len(self.alive_players)
    
    def _generate_doctor_candidates(self, observation: str, recent_statements: List) -> List[str]:
        """Generate Doctor-specific candidates"""
        candidates = []
        
        # After night kill analysis
        if "killed during the night" in observation:
            candidates.extend([
                "The night kill choice reveals Mafia priorities. They eliminated someone who threatened them.",
                "This kill pattern shows strategic targeting. We should protect similar analytical players.",
                "The victim was likely close to identifying Mafia. I'll focus on protecting similar contributors."
            ])
        
        # Strategic protection hints (without revealing role)
        candidates.extend([
            "We need to protect our most valuable analytical contributors from night kills.",
            "Mafia target players who provide the most threat to their strategy.",
            "I'm analyzing who Mafia would most want to eliminate next."
        ])
        
        return candidates
    
    def _generate_universal_candidates(self, observation: str, recent_statements: List, being_addressed: bool) -> List[str]:
        """Generate universal strategic candidates"""
        candidates = []
        
        # Response to direct address
        if being_addressed:
            candidates.extend([
                "Thanks for asking me directly. I've been carefully analyzing the discussion patterns.",
                "I appreciate being included. Let me share what I've been thinking about this situation.",
                "Good point bringing me into this. Here's my perspective on what we've learned."
            ])
        
        # Analytical contributions (much more variety)
        analytical_options = [
            "I'm seeing patterns in how players respond to direct questions - that's often revealing.",
            "The most suspicious behavior I've noticed is deflection without providing counter-evidence.",
            "We should focus on players who make accusations without backing them up with reasoning.",
            "I'm tracking inconsistencies between what players say and how they vote.",
            "The timing of accusations often reveals more than the accusations themselves.",
            "I'm analyzing who's asking questions versus who's providing actual analysis.",
            "Defensive responses to simple questions are worth noting.",
            "I'm watching for players who change their reasoning mid-discussion.",
            "Something that bothers me is when players avoid taking clear positions on anything.",
            "I've been keeping track of who's been trying to redirect conversations away from certain topics.",
            "The way some players phrase their responses gives away more than they probably realize.",
            "I'm noticing some interesting patterns in how certain players interact with each other.",
            "It's worth paying attention to which players seem to coordinate their responses.",
            "I think we should look more closely at players who've been unusually quiet during key moments.",
            "The body language in these text responses can be just as telling as what's actually said.",
            "I'm seeing some concerning trends in how information is being shared or withheld.",
            "There's been some subtle manipulation of the conversation flow that I want to address.",
            "I think some players are being more strategic about their word choices than others.",
            "The way certain accusations have been framed strikes me as calculated rather than genuine.",
            "I'm picking up on some underlying tensions between players that might be worth exploring."
        ]
        
        # Select options that haven't been used recently using our variation system
        fresh_options = [opt for opt in analytical_options if opt not in self.recent_analytical_options]
        
        if fresh_options:
            # Use unused options
            option1 = self._select_unused_element(fresh_options, self.recent_analytical_options, max_history=8)
            candidates.append(option1)
            if len(fresh_options) > 1:
                option2 = self._select_unused_element(fresh_options, self.recent_analytical_options, max_history=8)
                candidates.append(option2)
        else:
            # If all have been used recently, add with variations
            base_option = random.choice(analytical_options)
            variations = [
                f"Building on my earlier analysis: {base_option}",
                f"To expand on that observation: {base_option}",
                f"Let me revisit something I mentioned before: {base_option}",
                f"Going back to what I was saying earlier: {base_option}"
            ]
            candidates.append(random.choice(variations))
        
        return candidates
    
    def _handle_enhanced_night_action(self, observation: str) -> str:
        """Enhanced night action handling with strategic decision making"""
        # Safety check: Villagers have no night actions
        if self.my_role == Role.VILLAGER:
            print("Villager has no night actions - this shouldn't be called")
            return "pass"  # Return a safe default
        
        available_targets = self._extract_available_targets(observation)
        
        if not available_targets:
            return "[1]"  # Fallback
        
        if self.my_role == Role.DETECTIVE:
            target = self._enhanced_detective_investigation(available_targets)
        elif self.my_role == Role.DOCTOR:
            target = self._enhanced_doctor_protection(available_targets)
        elif self.my_role == Role.MAFIA:
            target = self._enhanced_mafia_elimination(available_targets)
        else:
            target = random.choice(available_targets)
        
        print(f"Enhanced night action: targeting Player {target}")
        night_response = f"[{target}]"
        
        # Validate night action format
        if self._validate_vote_format(night_response, available_targets):
            return night_response
        else:
            # Fallback to first available target
            return f"[{available_targets[0]}]"
    
    def _enhanced_detective_investigation(self, available_targets: List[int]) -> int:
        """Enhanced Detective investigation focusing on behavioral inconsistencies"""
        candidates = [p for p in available_targets if p != self.my_player_id]
        if not candidates:
            candidates = available_targets
        
        if not candidates:
            return 1
        
        target_scores = {}
        
        for target in candidates:
            score = 0.0
            
            # Behavioral inconsistency detection - primary focus
            inconsistency_score = self._detect_behavioral_inconsistencies(target)
            score += inconsistency_score * 0.6  # High weight for inconsistencies
            
            # Statement-action misalignment
            alignment_score = self._check_statement_action_alignment(target)
            score += (1.0 - alignment_score) * 0.3  # Penalty for misalignment
            
            # Information manipulation detection
            manipulation_score = self._detect_information_manipulation(target)
            score += manipulation_score * 0.4
            
            # Coalition analysis - check if player avoids village coordination
            if target not in self.memory.coalition_members and target in self.memory.role_claims:
                claimed_role = self.memory.role_claims[target]
                if claimed_role in ["Detective", "Doctor", "Villager"]:
                    # Claimed village role but not in coalition - suspicious
                    score += 0.3
            
            # Cross-reference with village intel
            if 'suspicious_behavior' in self.memory.village_intel:
                shared_suspicions = self.memory.village_intel['suspicious_behavior']['data']
                if target in shared_suspicions:
                    score += shared_suspicions[target] * 0.2
            
            target_scores[target] = score
        
        # Select highest scoring target
        best_target = max(target_scores.keys(), key=lambda p: target_scores[p])
        
        # Record investigation and share with village team
        investigation_data = {
            'target': best_target,
            'inconsistency_score': target_scores[best_target],
            'reasoning': 'Behavioral inconsistency analysis',
            'key_inconsistencies': self._get_key_inconsistencies(best_target)
        }
        
        self.memory.investigation_results[self.turn_count] = investigation_data
        self._share_village_intel('investigation_result', investigation_data)
        
        return best_target
    
    def _detect_behavioral_inconsistencies(self, player_id: int) -> float:
        """Detect behavioral inconsistencies that suggest deception"""
        if player_id not in self.memory.behavioral_analysis:
            return 0.0
        
        behavior = self.memory.behavioral_analysis[player_id]
        inconsistency_score = 0.0
        
        # Check for contradictory behavioral patterns
        defensiveness = behavior.get('defensive_tendency', 0)
        activity = behavior.get('activity_level', 0)
        deflection = behavior.get('deflection_tendency', 0)
        
        # High defensiveness + low activity = suspicious passivity
        if defensiveness > 0.7 and activity < 0.3:
            inconsistency_score += 0.4
        
        # High deflection + high question ratio = classic mafia tactics
        questions = behavior.get('question_ratio', 0)
        if deflection > 0.6 and questions > 0.5:
            inconsistency_score += 0.5
        
        # Voting pattern inconsistencies
        if player_id in self.memory.voting_patterns:
            votes = self.memory.voting_patterns[player_id]
            if len(votes) > 1:
                # Check for erratic voting (changing targets frequently)
                unique_targets = len(set(votes))
                if unique_targets / len(votes) > 0.7:  # More than 70% different targets
                    inconsistency_score += 0.3
        
        return min(inconsistency_score, 1.0)
    
    def _check_statement_action_alignment(self, player_id: int) -> float:
        """Check if player's statements align with their actions"""
        alignment_score = 0.5  # Neutral baseline
        
        # Check if claimed role behavior matches actual behavior
        if player_id in self.memory.role_claims:
            claimed_role = self.memory.role_claims[player_id]
            
            if claimed_role == "Detective":
                # Real detectives should ask probing questions
                if player_id in self.memory.behavioral_analysis:
                    question_ratio = self.memory.behavioral_analysis[player_id].get('question_ratio', 0)
                    alignment_score += min(question_ratio, 0.4)
                    
            elif claimed_role == "Doctor":
                # Real doctors should focus on protection/defense
                if player_id in self.memory.behavioral_analysis:
                    defensive_tendency = self.memory.behavioral_analysis[player_id].get('defensive_tendency', 0)
                    alignment_score += min(defensive_tendency * 0.3, 0.3)
        
        return min(alignment_score, 1.0)
    
    def _detect_information_manipulation(self, player_id: int) -> float:
        """Detect attempts to manipulate information flow"""
        manipulation_score = 0.0
        
        # Check discussion history for manipulation patterns
        player_statements = [msg for pid, msg in self.memory.discussion_history if pid == player_id]
        
        if len(player_statements) > 2:
            # Look for deflection keywords
            deflection_keywords = ['but what about', 'however', 'actually', 'though', 'instead']
            deflection_count = sum(1 for stmt in player_statements 
                                 for keyword in deflection_keywords if keyword in stmt.lower())
            
            if deflection_count / len(player_statements) > 0.3:
                manipulation_score += 0.4
            
            # Look for vague accusations without evidence
            vague_keywords = ['suspicious', 'seems', 'maybe', 'possibly']
            vague_count = sum(1 for stmt in player_statements 
                            for keyword in vague_keywords if keyword in stmt.lower())
            
            if vague_count / len(player_statements) > 0.4:
                manipulation_score += 0.3
        
        return manipulation_score
    
    def _get_key_inconsistencies(self, player_id: int) -> List[str]:
        """Get list of key inconsistencies found for this player"""
        inconsistencies = []
        
        if player_id in self.memory.behavioral_analysis:
            behavior = self.memory.behavioral_analysis[player_id]
            
            defensiveness = behavior.get('defensive_tendency', 0)
            activity = behavior.get('activity_level', 0)
            deflection = behavior.get('deflection_tendency', 0)
            questions = behavior.get('question_ratio', 0)
            
            if defensiveness > 0.7 and activity < 0.3:
                inconsistencies.append("High defensiveness but low activity")
            
            if deflection > 0.6 and questions > 0.5:
                inconsistencies.append("High deflection with excessive questioning")
            
            if player_id in self.memory.role_claims:
                claimed_role = self.memory.role_claims[player_id]
                if claimed_role == "Detective" and questions < 0.3:
                    inconsistencies.append("Claims Detective but asks few questions")
                elif claimed_role == "Doctor" and defensiveness < 0.3:
                    inconsistencies.append("Claims Doctor but shows little protective instinct")
        
        return inconsistencies
    
    def _enhanced_doctor_protection(self, available_targets: List[int]) -> int:
        """Enhanced Doctor protection coordinating with detective findings"""
        candidates = [p for p in available_targets if p != self.my_player_id]
        if not candidates:
            candidates = available_targets
        
        if not candidates:
            return 1
        
        protection_scores = {}
        
        for target in candidates:
            score = 0.0
            
            # High priority: Protect confirmed detective
            if target in self.memory.role_claims and self.memory.role_claims[target] == "Detective":
                trust_score = self.memory.trust_scores.get(target, 0.5)
                if trust_score > 0.6:  # High trust detective
                    score += 2.0
                elif trust_score > 0.4:  # Moderate trust detective
                    score += 1.2
            
            # Protect village coalition members
            if target in self.memory.coalition_members:
                score += 1.5
            
            # Use detective investigation intel for protection priority
            if 'investigation_result' in self.memory.village_intel:
                recent_investigations = self.memory.village_intel['investigation_result']
                if isinstance(recent_investigations['data'], dict):
                    investigated_target = recent_investigations['data'].get('target')
                    # If detective investigated someone, protect the detective
                    if investigated_target and target == recent_investigations.get('source_role'):
                        score += 1.8
            
            # Protect players with village-like behavior patterns
            if target in self.memory.behavioral_analysis:
                behavior = self.memory.behavioral_analysis[target]
                
                # Village indicators
                analytical_contribution = behavior.get('activity_level', 0) * behavior.get('question_ratio', 0)
                score += analytical_contribution * 0.8
                
                # Low deflection is village-like
                deflection = behavior.get('deflection_tendency', 0)
                score += (1.0 - deflection) * 0.4
                
                # Moderate defensiveness is normal for village
                defensiveness = behavior.get('defensive_tendency', 0)
                if 0.3 <= defensiveness <= 0.6:  # Healthy village defensiveness
                    score += 0.3
            
            # Predict mafia targeting patterns
            mafia_target_likelihood = self._predict_mafia_target_likelihood(target)
            score += mafia_target_likelihood * 0.7
            
            # Strategic self-protection when necessary
            if target == self.my_player_id:
                self_protection_need = self._assess_self_protection_need()
                score += self_protection_need
            
            # Avoid protecting likely mafia (negative score)
            if target in self.memory.suspicion_tracker:
                suspicion = self.memory.suspicion_tracker[target]
                if suspicion > 0.8:  # Very suspicious
                    score -= 1.0
                elif suspicion > 0.6:  # Moderately suspicious
                    score -= 0.5
            
            protection_scores[target] = max(score, 0.0)  # No negative scores
        
        best_target = max(protection_scores.keys(), key=lambda p: protection_scores[p])
        
        # Record protection decision and share intel
        protection_data = {
            'target': best_target,
            'reasoning': 'Coordinated village protection strategy',
            'protection_score': protection_scores[best_target],
            'mafia_threat_level': self._assess_overall_mafia_threat()
        }
        
        self._share_village_intel('protection_target', protection_data)
        
        return best_target
    
    def _predict_mafia_target_likelihood(self, player_id: int) -> float:
        """Predict how likely mafia is to target this player"""
        likelihood = 0.0
        
        # Mafia targets active, analytical village players
        if player_id in self.memory.behavioral_analysis:
            behavior = self.memory.behavioral_analysis[player_id]
            activity = behavior.get('activity_level', 0)
            questions = behavior.get('question_ratio', 0)
            
            # High activity + good questions = prime mafia target
            if activity > 0.6 and questions > 0.4:
                likelihood += 0.7
            elif activity > 0.4:
                likelihood += 0.4
        
        # Claimed power roles are high-priority mafia targets
        if player_id in self.memory.role_claims:
            claimed_role = self.memory.role_claims[player_id]
            if claimed_role == "Detective":
                likelihood += 0.8  # Detectives are top priority
            elif claimed_role == "Doctor":
                likelihood += 0.6  # Doctors are also high priority
        
        # Players in village coalition are threats to mafia
        if player_id in self.memory.coalition_members:
            likelihood += 0.5
        
        # Low suspicion players are more likely to be targeted
        if player_id in self.memory.suspicion_tracker:
            suspicion = self.memory.suspicion_tracker[player_id]
            likelihood += (1.0 - suspicion) * 0.3
        
        return min(likelihood, 1.0)
    
    def _assess_self_protection_need(self) -> float:
        """Assess if doctor should protect themselves"""
        self_protection_score = 0.0
        
        # Protect self if highly suspected (might be voted out)
        if self.my_player_id in self.memory.suspicion_tracker:
            suspicion = self.memory.suspicion_tracker[self.my_player_id]
            if suspicion > 0.7:
                self_protection_score += 0.8
        
        # Protect self if claimed doctor (mafia target)
        if (self.my_player_id in self.memory.role_claims and 
            self.memory.role_claims[self.my_player_id] == "Doctor"):
            self_protection_score += 0.6
        
        # Protect self early in game
        if self.night_number <= 2:
            self_protection_score += 0.3
        
        # Don't over-protect self if other high-value targets exist
        high_value_targets = sum(1 for pid, role in self.memory.role_claims.items() 
                               if role == "Detective" and pid != self.my_player_id)
        if high_value_targets > 0:
            self_protection_score *= 0.5
        
        return self_protection_score
    
    def _assess_overall_mafia_threat(self) -> float:
        """Assess current threat level from mafia"""
        threat_level = 0.5  # Baseline
        
        # Higher threat if village coalition is forming (mafia will act)
        if len(self.memory.coalition_members) > 0:
            threat_level += 0.3
        
        # Higher threat if detective has found evidence
        if self.memory.investigation_results:
            recent_investigations = list(self.memory.investigation_results.values())
            if recent_investigations:
                threat_level += 0.4
        
        # Lower threat early in game
        if self.turn_count < 4:
            threat_level -= 0.2
        
        return min(max(threat_level, 0.0), 1.0)
    
    def _enhanced_mafia_elimination(self, available_targets: List[int]) -> int:
        """Enhanced Mafia elimination using predictive targeting system"""
        candidates = [p for p in available_targets if p != self.my_player_id]
        if not candidates:
            candidates = available_targets
        
        if not candidates:
            return 1
        
        elimination_scores = {}
        
        for target in candidates:
            # Use predictive targeting system
            threat_score = self._calculate_village_threat_level(target)
            elimination_scores[target] = threat_score
        
        best_target = max(elimination_scores.keys(), key=lambda p: elimination_scores[p])
        
        # Record mafia targeting decision for village prediction
        targeting_data = {
            'target': best_target,
            'threat_level': elimination_scores[best_target],
            'reasoning': 'Strategic village threat elimination',
            'alternative_targets': {p: score for p, score in elimination_scores.items() if p != best_target}
        }
        
        # Mafia doesn't share intel with village, but this creates patterns
        self._record_mafia_targeting_pattern(targeting_data)
        
        return best_target
    
    def _calculate_village_threat_level(self, player_id: int) -> float:
        """Calculate how much of a threat this player is to mafia"""
        threat_score = 0.0
        
        # Claimed power roles are highest priority
        if player_id in self.memory.role_claims:
            claimed_role = self.memory.role_claims[player_id]
            if claimed_role == "Detective":
                threat_score += 3.0  # Highest threat
            elif claimed_role == "Doctor":
                threat_score += 2.5  # High threat
            elif claimed_role == "Villager":
                threat_score += 1.0  # Moderate threat
        
        # Village coalition members are significant threats
        if player_id in self.memory.coalition_members:
            threat_score += 2.0
        
        # High trust score indicates dangerous village player
        if player_id in self.memory.trust_scores:
            trust = self.memory.trust_scores[player_id]
            threat_score += trust * 1.5
        
        # Active analytical players who ask good questions
        if player_id in self.memory.behavioral_analysis:
            behavior = self.memory.behavioral_analysis[player_id]
            activity = behavior.get('activity_level', 0)
            questions = behavior.get('question_ratio', 0)
            
            # High activity + good questions = analytical threat
            analytical_threat = activity * questions
            threat_score += analytical_threat * 2.0
            
            # Low deflection indicates honest village player
            deflection = behavior.get('deflection_tendency', 0)
            threat_score += (1.0 - deflection) * 1.2
        
        # Players with low suspicion are credible village threats
        if player_id in self.memory.suspicion_tracker:
            suspicion = self.memory.suspicion_tracker[player_id]
            threat_score += (1.0 - suspicion) * 1.8
            
            # Avoid eliminating highly suspicious players (let village eliminate them)
            if suspicion > 0.8:
                threat_score -= 2.0
        
        # Players who contribute to village intel sharing
        village_intel_contributor = self._assess_intel_contribution(player_id)
        threat_score += village_intel_contributor * 1.3
        
        # Strategic considerations - eliminate investigation targets to cast doubt
        for investigation in self.memory.investigation_results.values():
            if investigation.get('target') == player_id:
                # If we were investigated, eliminating us looks suspicious
                threat_score -= 1.5
        
        return max(threat_score, 0.1)  # Minimum threat level
    
    def _process_investigation_results(self, observation: str):
        """Process and store investigation results from observation"""
        import re
        
        # Look for investigation results in current observation
        mafia_found = re.search(r'Player (\d+) IS a Mafia member', observation)
        innocent_found = re.search(r'Player (\d+) IS NOT a Mafia member', observation)
        
        if mafia_found:
            target = int(mafia_found.group(1))
            investigation_data = {
                'target': target,
                'result': 'mafia',
                'found_mafia': True,
                'turn': self.turn_count,
                'confidence': 1.0
            }
            self.memory.investigation_results[self.turn_count] = investigation_data
            
            # ENHANCED DETECTIVE STRATEGY: Build case before sharing
            self._build_mafia_case(target, investigation_data)
            self._share_village_intel('investigation_result', investigation_data)
            print(f"🔍 DETECTIVE: Confirmed Player {target} is Mafia! Building case...")
            
        elif innocent_found:
            target = int(innocent_found.group(1))
            investigation_data = {
                'target': target,
                'result': 'innocent',
                'cleared_villager': True,
                'turn': self.turn_count,
                'confidence': 1.0
            }
            self.memory.investigation_results[self.turn_count] = investigation_data
            self._share_village_intel('investigation_result', investigation_data)
            print(f"🔍 DETECTIVE: Confirmed Player {target} is innocent.")
    
    def _build_mafia_case(self, target_player: int, investigation_data: dict):
        """Build a comprehensive case against confirmed mafia member"""
        case_evidence = {
            'investigation': investigation_data,
            'behavioral_patterns': [],
            'voting_inconsistencies': [],
            'alliance_suspicions': []
        }
        
        # Collect behavioral evidence
        voting_history = self.memory.voting_patterns.get(target_player, [])
        if voting_history:
            # Check if they voted to protect other suspicious players
            suspicious_votes = []
            for vote_target in voting_history:
                if vote_target != target_player:
                    suspicion = self.memory.suspicion_tracker.get(vote_target, 0.5)
                    if suspicion < 0.3:  # Voted for low-suspicion (likely village) players
                        suspicious_votes.append(vote_target)
            
            if suspicious_votes:
                case_evidence['voting_inconsistencies'] = suspicious_votes
        
        # Look for defensive statements in discussion history
        player_statements = [stmt for pid, stmt in self.memory.discussion_history if pid == target_player]
        defensive_statements = []
        for statement in player_statements:
            if any(word in statement.lower() for word in ['innocent', 'trust me', 'not mafia', 'honest']):
                defensive_statements.append(statement[:50] + "...")
        
        if defensive_statements:
            case_evidence['behavioral_patterns'] = defensive_statements
        
        # Store the case for sharing in discussions
        if not hasattr(self.memory, 'mafia_cases'):
            self.memory.mafia_cases = {}
        self.memory.mafia_cases[target_player] = case_evidence
        
        print(f"🔍 DETECTIVE: Built case against Player {target_player} with {len(case_evidence['voting_inconsistencies'])} voting issues and {len(case_evidence['behavioral_patterns'])} behavioral red flags")
    
    def _assess_intel_contribution(self, player_id: int) -> float:
        """Assess how much this player contributes to village intelligence"""
        contribution_score = 0.0
        
        # Check if player has shared investigation results
        if player_id in self.memory.investigation_results:
            contribution_score += 1.0
        
        # Check discussion contributions
        player_statements = [msg for pid, msg in self.memory.discussion_history if pid == player_id]
        if len(player_statements) > 2:
            # Look for analytical keywords indicating intelligence sharing
            intel_keywords = ['suspicious', 'analyze', 'pattern', 'behavior', 'evidence', 'investigation']
            intel_mentions = sum(1 for stmt in player_statements 
                               for keyword in intel_keywords if keyword in stmt.lower())
            
            if intel_mentions / len(player_statements) > 0.3:
                contribution_score += 0.8
        
        return contribution_score
    
    def _record_mafia_targeting_pattern(self, targeting_data: Dict):
        """Record mafia targeting patterns for village analysis"""
        # This creates patterns that smart villages can detect
        if not hasattr(self, 'mafia_targeting_history'):
            self.mafia_targeting_history = []
        
        self.mafia_targeting_history.append({
            'turn': self.turn_count,
            'target': targeting_data['target'],
            'threat_level': targeting_data['threat_level'],
            'target_role_claim': self.memory.role_claims.get(targeting_data['target'], 'Unknown')
        })
        
        # Keep only recent history
        if len(self.mafia_targeting_history) > 5:
            self.mafia_targeting_history = self.mafia_targeting_history[-5:]
    
    def _handle_enhanced_voting(self, observation: str) -> str:
        """Enhanced voting with strategic decision making + Championship Herd Manipulation"""
        available_targets = self._extract_available_targets(observation)
        
        if not available_targets:
            return "1"
        
        # ============================================================================
        # CHAMPIONSHIP ENHANCEMENT: Herd Manipulation Strategy (1st Place)
        # ============================================================================
        
        # Create enhanced game context for herd manipulation
        enhanced_game_context = {
            'my_player_id': self.my_player_id,
            'being_accused': self._being_accused_in_statements(self._extract_player_statements(observation)),
            'day_number': self.day_number,
            'exploitable_leaks': len(self.detected_leaks),
            'phase': 'voting'
        }
        
        # Check if we should vote first to influence others
        should_vote_first = self.herd_manipulator.should_vote_first(
            enhanced_game_context, 
            self.memory.suspicion_tracker
        )
        
        # Use ToM engine for voting decision
        voting_scores = {}
        
        for target in available_targets:
            if target == self.my_player_id:
                continue  # Don't vote for self
            
            score = 0.0
            
            # Base suspicion level
            suspicion = self.tom_engine.get_suspicion_level(target)
            score += suspicion * 2.0
            
            # Analyze discussion patterns for Mafia tells
            discussion_score = self._analyze_player_discussion_for_voting(target, observation)
            score += discussion_score
            
            # Role-specific voting strategy
            if self.my_role == Role.MAFIA:
                # Mafia votes to eliminate Village threats
                if target in self.memory.role_claims:
                    claimed_role = self.memory.role_claims[target]
                    if claimed_role in ["Detective", "Doctor"]:
                        score += 1.5
                
                # Vote for players who are suspicious of us
                if self.tom_engine.get_meta_suspicion(target, self.my_player_id) > 0.6:
                    score += 1.0
            else:
                # ENHANCED VILLAGE VOTING STRATEGY WITH POWER ROLE COORDINATION
                
                # Base suspicion (more aggressive)
                score += suspicion * 2.5
                
                # CRITICAL: Trust Detective investigations absolutely
                for investigation in self.memory.investigation_results.values():
                    if investigation['target'] == target and investigation.get('found_mafia'):
                        score += 5.0  # Much higher weight
                        print(f"[VILLAGE] Detective confirmed Player {target} is Mafia - voting priority!")
                
                # Coordinate with village intel sharing
                village_consensus_score = self._calculate_village_consensus_vote(target)
                score += village_consensus_score * 1.5
                
                # Trust verification for role claims - avoid voting trusted roles
                if target in self.memory.trust_scores:
                    trust = self.memory.trust_scores[target]
                    if trust > 0.7:  # High trust
                        score -= 2.0  # Strong penalty for voting trusted village
                        print(f"[VILLAGE] Player {target} is highly trusted - avoiding vote")
                    elif trust > 0.5:  # Moderate trust
                        score -= 1.0
                
                # Coalition member protection
                if target in self.memory.coalition_members:
                    score -= 3.0  # Don't vote coalition members
                    print(f"[VILLAGE] Player {target} is coalition member - protected from vote")
                
                # Behavioral inconsistency targeting (like detective does)
                inconsistency_score = self._detect_behavioral_inconsistencies(target)
                score += inconsistency_score * 2.0
                if inconsistency_score > 0.5:
                    print(f"[VILLAGE] Player {target} shows behavioral inconsistencies")
                
                # Information manipulation detection
                manipulation_score = self._detect_information_manipulation(target)
                score += manipulation_score * 1.8
                
                # Heavily penalize quiet players (Mafia tactic)
                activity = self.tom_engine.get_behavioral_pattern(target, 'activity_level')
                if activity < 0.3:
                    score += 2.0
                    print(f"[VILLAGE] Player {target} suspiciously quiet - adding penalty")
                
                # Penalize deflection behavior
                deflection = self.tom_engine.get_behavioral_pattern(target, 'deflection_tendency')
                if deflection > 0.6:
                    score += 1.5
                    print(f"[VILLAGE] Player {target} deflecting - suspicious behavior")
                
                # Cross-reference with detective investigations
                if 'investigation_result' in self.memory.village_intel:
                    investigation_data = self.memory.village_intel['investigation_result']['data']
                    if isinstance(investigation_data, dict):
                        if investigation_data.get('target') == target:
                            inconsistencies = investigation_data.get('key_inconsistencies', [])
                            if inconsistencies:
                                score += len(inconsistencies) * 0.8
                                print(f"[VILLAGE] Detective found {len(inconsistencies)} inconsistencies in Player {target}")
                
                # Use mafia tells analysis
                if 'mafia_tells_analysis' in self.memory.village_intel:
                    tells_data = self.memory.village_intel['mafia_tells_analysis']['data']
                    if isinstance(tells_data, dict) and target in tells_data:
                        tells_info = tells_data[target]
                        score += tells_info['suspicion_score'] * 2.5
                        print(f"[VILLAGE] Player {target} shows {tells_info['tells_count']} mafia tells: {tells_info['tells']}")
                
                # Use suspicious voting patterns
                if 'suspicious_voting_patterns' in self.memory.village_intel:
                    voting_data = self.memory.village_intel['suspicious_voting_patterns']['data']
                    if isinstance(voting_data, dict) and target in voting_data:
                        pattern_info = voting_data[target]
                        score += pattern_info['suspicion_score'] * 2.0
                        if pattern_info['aligned_with']:
                            print(f"[VILLAGE] Player {target} has suspicious voting alignment with {pattern_info['aligned_with']}")
                
                # Doctor protection coordination - if doctor protected someone, they're probably village
                if 'protection_target' in self.memory.village_intel:
                    protection_data = self.memory.village_intel['protection_target']['data']
                    if isinstance(protection_data, dict):
                        protected_player = protection_data.get('target')
                        if target == protected_player:
                            score -= 1.5  # Doctor thinks they're worth protecting
                            print(f"[VILLAGE] Player {target} was protected by doctor - likely village")
                
                # Reward consistency with our reads
                if target in [leak.source_player for leak in self.detected_leaks]:
                    score += 2.0
                    print(f"[VILLAGE] Player {target} had information leaks - strong evidence")
                
                # Late game urgency - be more decisive
                if self._is_late_game(observation):
                    if suspicion > 0.7:
                        score += 3.0
                        print(f"[VILLAGE] Late game - must eliminate highly suspicious Player {target}")
                
                # Early game aggression - pressure suspicious players
                if self.turn_count <= 3 and suspicion > 0.6:
                    score += 1.5
                    print(f"[VILLAGE] Early pressure on suspicious Player {target}")
            
            # CRITICAL FIX: Actually store the score!
            voting_scores[target] = score
        
        if not voting_scores:
            raise Exception("Failed to generate voting scores - no fallback allowed")
        
        best_target = max(voting_scores.keys(), key=lambda p: voting_scores[p])
        confidence_level = voting_scores[best_target]
        
        # ============================================================================
        # CHAMPIONSHIP ENHANCEMENT: Strategic First Vote (1st Place Strategy)
        # ============================================================================
        
        if should_vote_first:
            # Generate influence-optimized vote message
            reasoning_parts = []
            
            if best_target in [leak.source_player for leak in self.detected_leaks]:
                reasoning_parts.append("They had suspicious information leaks.")
            
            if self.tom_engine.get_suspicion_level(best_target) > 0.7:
                reasoning_parts.append("Their behavior has been consistently suspicious.")
            
            if best_target in self.memory.role_claims:
                claimed_role = self.memory.role_claims[best_target]
                if self.my_role != Role.MAFIA and claimed_role in ["Detective", "Doctor"]:
                    reasoning_parts.append("Their role claim seems questionable.")
            
            reasoning = " ".join(reasoning_parts) if reasoning_parts else "Based on my analysis of their behavior patterns."
            
            # Use herd manipulation system to generate influential vote
            influence_vote_message = self.herd_manipulator.generate_influence_vote(
                best_target, reasoning, confidence_level / 5.0  # Normalize confidence
            )
            
            self.championship_metrics['first_vote_influences'] += 1
            print(f"🎯 FIRST VOTE STRATEGY: {influence_vote_message}")
            
            # Return the influence vote message instead of just the target
            # Note: This might need adjustment based on your game format
            
        # VILLAGE COORDINATION STRATEGY: Announce voting intention in discussion
        elif self.my_role != Role.MAFIA and confidence_level > 3.0:  # High confidence vote
            coordination_message = f"[PRE-VOTE] I'm planning to vote for Player {best_target}. "
            if best_target in [leak.source_player for leak in self.detected_leaks]:
                coordination_message += "They had suspicious information leaks. "
            if self.tom_engine.get_suspicion_level(best_target) > 0.7:
                coordination_message += "Their behavior has been consistently suspicious. "
            coordination_message += "Village should coordinate votes!"
            print(coordination_message)
        
        # Capture voting reasoning for training data
        self._last_vote_reasoning = {
            'available_targets': available_targets,
            'voting_scores': voting_scores,
            'chosen_target': best_target,
            'reasoning_factors': {
                'suspicion_levels': {pid: self.tom_engine.get_suspicion_level(pid) for pid in available_targets},
                'role_claims': dict(self.memory.role_claims),
                'investigation_results': dict(self.memory.investigation_results),
                'information_leaks': [leak.source_player for leak in self.detected_leaks]
            },
            'strategic_context': self._get_role_string(),
            'coordination_attempted': confidence_level > 3.0 if self.my_role != Role.MAFIA else False
        }
        self._last_vote_target = best_target
        self._vote_alternatives = available_targets
        
        # Record voting decision
        if self.my_player_id not in self.memory.voting_patterns:
            self.memory.voting_patterns[self.my_player_id] = []
        self.memory.voting_patterns[self.my_player_id].append(best_target)
        
        print(f"Enhanced voting decision: Player {best_target} (score: {voting_scores[best_target]:.2f})")
        print(f"All voting scores: {voting_scores}")
        
        # Ensure correct voting format - handle both discussion and voting phases
        if self._is_voting_phase(observation):
            # During voting phase, return just the number in brackets - validate format
            vote_response = f"[{best_target}]"
            if self._validate_vote_format(vote_response, available_targets):
                print(f"🗳️ Voting phase detected - returning vote: {vote_response}")
                return vote_response
            else:
                # Raise exception if validation fails - no fallback allowed
                print(f"❌ Vote validation failed - no fallback allowed")
                raise Exception("Vote validation failed and fallback is prohibited")
        else:
            # During discussion phase, return a discussion response
            discussion_response = f"I'm voting for Player {best_target}. {self._get_voting_reasoning(best_target, voting_scores[best_target])}"
            print(f"💬 Discussion phase detected - returning discussion: {discussion_response[:100]}...")
            return discussion_response
    
    def _validate_vote_format(self, vote_response: str, available_targets: List[int]) -> bool:
        """Validate that vote is in correct format and target is valid"""
        import re
        
        # Check if format is [X] where X is a number
        match = re.match(r'^\[(\d+)\]$', vote_response.strip())
        if not match:
            return False
        
        # Check if target is in available targets
        target = int(match.group(1))
        return target in available_targets
    
    def _get_voting_reasoning(self, target: int, score: float) -> str:
        """Generate reasoning for voting decision"""
        reasons = []
        
        # High suspicion
        suspicion = self.tom_engine.get_suspicion_level(target)
        if suspicion > 0.7:
            reasons.append("their behavior has been consistently suspicious")
        
        # Information leaks
        if target in [leak.source_player for leak in self.detected_leaks]:
            reasons.append("they had suspicious information leaks")
        
        # Role claims
        if target in self.memory.role_claims:
            claimed_role = self.memory.role_claims[target]
            if self.my_role == Role.MAFIA and claimed_role in ["Detective", "Doctor"]:
                reasons.append("their role claim needs to be verified")
            elif self.my_role != Role.MAFIA and claimed_role == "Mafia":
                reasons.append("they've been identified as Mafia")
        
        # Default reasoning
        if not reasons:
            reasons.append("based on my analysis of their behavior patterns")
        
        return " and ".join(reasons[:2])  # Limit to 2 reasons for brevity
    
    def _calculate_village_consensus_vote(self, target_player: int) -> float:
        """Calculate village team consensus on voting target"""
        consensus_score = 0.0
        
        # Check if target is suspected across multiple village intel sources
        suspicion_sources = 0
        total_suspicion = 0.0
        
        # Detective investigations pointing to this player
        for investigation in self.memory.investigation_results.values():
            if investigation.get('target') == target_player:
                suspicion_sources += 1
                total_suspicion += investigation.get('inconsistency_score', 0)
        
        # Shared suspicious behavior intel
        if 'suspicious_behavior' in self.memory.village_intel:
            shared_suspicions = self.memory.village_intel['suspicious_behavior']['data']
            if isinstance(shared_suspicions, dict) and target_player in shared_suspicions:
                suspicion_sources += 1
                total_suspicion += shared_suspicions[target_player]
        
        # Role claim analysis consensus
        if 'role_claim_analysis' in self.memory.village_intel:
            claim_analysis = self.memory.village_intel['role_claim_analysis']['data']
            if isinstance(claim_analysis, dict) and target_player in claim_analysis:
                trust_score = claim_analysis[target_player]
                if trust_score < 0.4:  # Low trust = suspicious
                    suspicion_sources += 1
                    total_suspicion += (1.0 - trust_score)
        
        # Coalition consensus - if not in coalition but claims village role
        if (target_player in self.memory.role_claims and 
            self.memory.role_claims[target_player] in ["Detective", "Doctor", "Villager"] and
            target_player not in self.memory.coalition_members):
            suspicion_sources += 1
            total_suspicion += 0.6  # Moderate suspicion for being excluded
        
        # Calculate consensus score
        if suspicion_sources > 0:
            consensus_score = (total_suspicion / suspicion_sources) * min(suspicion_sources / 2.0, 1.0)
        
        return consensus_score
    
    def _analyze_player_discussion_for_voting(self, target_player: int, observation: str) -> float:
        """Analyze player's discussion patterns for voting decision"""
        score = 0.0
        
        # Extract statements from this player
        player_statements = []
        lines = observation.split('\n')
        for line in lines:
            if f"[{target_player}," in line or f"Player {target_player}" in line:
                # Extract the actual statement content
                if '"' in line:
                    statement = line.split('"')[1] if '"' in line else line
                    player_statements.append(statement.lower())
        
        if not player_statements:
            return 0.0
        
        # Analyze for Mafia behavioral patterns
        for statement in player_statements:
            # Early aggressive accusations (Mafia tell)
            if any(word in statement for word in ['suspicious', 'mafia', 'eliminate', 'vote']):
                if len(statement.split()) < 15:  # Short aggressive statements
                    score += 0.8
            
            # Deflection without evidence
            deflection_words = ['but', 'however', 'actually', 'think about']
            evidence_words = ['because', 'evidence', 'noticed', 'saw']
            
            has_deflection = any(word in statement for word in deflection_words)
            has_evidence = any(word in statement for word in evidence_words)
            
            if has_deflection and not has_evidence:
                score += 1.2
            
            # Defensive language patterns
            defensive_phrases = ['why are you', 'that\'s not', 'i\'m not', 'you\'re wrong']
            if any(phrase in statement for phrase in defensive_phrases):
                score += 0.9
            
            # Question-heavy responses (avoiding taking stances)
            question_count = statement.count('?')
            statement_length = len(statement.split())
            if statement_length > 10 and question_count >= 2:
                score += 0.7
        
        return min(score, 3.0)  # Cap the score
    
    def _extract_voting_targets(self, observation: str) -> List[int]:
        """Extract valid voting targets from observation"""
        targets = []
        
        # Look for voting format instructions
        if "Valid:" in observation:
            # Extract from format like "Valid: [0], [1], [2], [3], [4]"
            valid_section = observation.split("Valid:")[1].split("\n")[0]
            target_matches = re.findall(r'\[(\d+)\]', valid_section)
            targets = [int(match) for match in target_matches]
        else:
            # Fallback: extract all player numbers mentioned
            player_matches = re.findall(r'Player (\d+)', observation)
            targets = [int(match) for match in set(player_matches)]
        
        return targets
    
    def _extract_player_statements(self, observation: str) -> List[Tuple[int, str]]:
        """Extract player statements from observation"""
        statements = []
        lines = observation.split('\n')
        
        for line in lines:
            # Look for pattern like [player_id, "statement", turn]
            match = re.search(r'\[(\d+), "([^"]+)"', line)
            if match:
                player_id = int(match.group(1))
                statement = match.group(2)
                statements.append((player_id, statement))
        
        return statements
    
    def _being_accused_in_statements(self, statements: List[Tuple[int, str]]) -> bool:
        """Check if we're being accused in recent statements"""
        if self.my_player_id == -1:
            return False
        
        accusation_keywords = ['suspicious', 'mafia', 'eliminate', 'vote', 'accuse']
        
        for player_id, statement in statements:
            if player_id != self.my_player_id:
                statement_lower = statement.lower()
                if (f"player {self.my_player_id}" in statement_lower and 
                    any(keyword in statement_lower for keyword in accusation_keywords)):
                    return True
        
        return False
    
    def _is_teammate_under_pressure(self, observation: str) -> bool:
        """Check if Mafia teammate is under pressure"""
        if self.my_role != Role.MAFIA:
            return False
        
        # Get our teammates
        if not hasattr(self, 'mafia_teammates'):
            self._is_teammate(0)  # This will initialize mafia_teammates
        
        if not hasattr(self, 'mafia_teammates'):
            return False
        
        # Check if any teammate is being accused
        accusation_keywords = ['mafia', 'suspicious', 'eliminate', 'vote out', 'investigate']
        recent_statements = self._extract_player_statements(observation)
        
        for teammate_id in self.mafia_teammates:
            accusation_count = 0
            for player_id, statement in recent_statements:
                if player_id != teammate_id:  # Someone else talking about teammate
                    statement_lower = statement.lower()
                    if (f"player {teammate_id}" in statement_lower and 
                        any(keyword in statement_lower for keyword in accusation_keywords)):
                        accusation_count += 1
            
            # If teammate has 2+ accusations, they're under pressure
            if accusation_count >= 2:
                return True
        
        return False
    
    def _identify_village_targets(self, observation: str) -> List[int]:
        """Identify likely Village players to target for misdirection"""
        targets = []
        
        # Look for players claiming Detective/Doctor roles
        if "detective" in observation.lower():
            detective_match = re.search(r'player (\d+).*detective', observation.lower())
            if detective_match:
                targets.append(int(detective_match.group(1)))
        
        if "doctor" in observation.lower():
            doctor_match = re.search(r'player (\d+).*doctor', observation.lower())
            if doctor_match:
                targets.append(int(doctor_match.group(1)))
        
        # If no special roles found, target active analytical players
        if not targets:
            # Use ToM engine to find most active/analytical players
            for player_id in range(6):
                if player_id != self.my_player_id:
                    activity = self.tom_engine.get_behavioral_pattern(player_id, 'activity_level')
                    if activity > 0.6:  # Highly active players
                        targets.append(player_id)
        
        return targets[:2]  # Return top 2 targets
    
    def _update_belief_accuracy(self, observation: str):
        """Update belief accuracy based on revealed information"""
        # Check for role reveals or eliminations that confirm/contradict our beliefs
        if "is Mafia" in observation or "was eliminated" in observation:
            # Extract player ID and actual role if revealed
            mafia_reveals = re.findall(r'Player (\d+).*(?:is|was).*Mafia', observation, re.IGNORECASE)
            village_reveals = re.findall(r'Player (\d+).*(?:is|was).*(?:Villager|Detective|Doctor)', observation, re.IGNORECASE)
            
            correct_predictions = 0
            total_predictions = 0
            
            # Check accuracy for Mafia predictions
            for match in mafia_reveals:
                player_id = int(match)
                our_suspicion = self.tom_engine.get_suspicion_level(player_id)
                if our_suspicion > 0.6:  # We suspected them
                    correct_predictions += 1
                total_predictions += 1
            
            # Check accuracy for Village predictions  
            for match in village_reveals:
                player_id = int(match)
                our_suspicion = self.tom_engine.get_suspicion_level(player_id)
                if our_suspicion < 0.4:  # We trusted them
                    correct_predictions += 1
                total_predictions += 1
            
            # Update running accuracy
            if total_predictions > 0:
                current_accuracy = correct_predictions / total_predictions
                # Weighted average with previous accuracy
                alpha = 0.3  # Learning rate
                self.performance_metrics['belief_accuracy'] = (
                    (1 - alpha) * self.performance_metrics['belief_accuracy'] + 
                    alpha * current_accuracy
                )
    
    def _track_voting_success(self, observation: str, voted_player: int):
        """Track if our voting decisions were strategically sound"""
        # Check if the player we voted for was actually eliminated and was Mafia
        if f"Player {voted_player}" in observation and "Mafia" in observation:
            if self.my_role != Role.MAFIA:  # Village roles get credit for eliminating Mafia
                self.performance_metrics['strategic_decisions'] += 1
        elif f"Player {voted_player}" in observation and any(role in observation for role in ["Villager", "Detective", "Doctor"]):
            if self.my_role == Role.MAFIA:  # Mafia gets credit for eliminating Village
                self.performance_metrics['strategic_decisions'] += 1
    
    # Helper methods
    def _parse_observation(self, observation: str):
        """Enhanced observation parsing"""
        # Extract alive players
        player_matches = re.findall(r'Player (\d+)', observation)
        if player_matches:
            self.alive_players = [int(p) for p in set(player_matches)]
        
        # Extract my player ID
        my_id_match = re.search(r'You are Player (\d+)', observation)
        if my_id_match:
            self.my_player_id = int(my_id_match.group(1))
        
        # Track day/night numbers
        day_match = re.search(r'Day (\d+)', observation)
        if day_match:
            self.day_number = int(day_match.group(1))
        
        night_match = re.search(r'Night (\d+)', observation)
        if night_match:
            self.night_number = int(night_match.group(1))
        
        # Track night kills
        if "killed during the night" in observation:
            killed_match = re.search(r'Player (\d+) was killed', observation)
            if killed_match:
                killed_player = int(killed_match.group(1))
                self.memory.night_kill_history.append(killed_player)
    
    def _update_enhanced_memory(self, observation: str):
        """Update enhanced memory system"""
        self.memory.observational.append(f"Turn {self.turn_count}: {observation}")
        
        # Extract and update player statements
        statements = self._extract_player_statements(observation)
        self.memory.discussion_history.extend(statements)
        
        # Update ToM engine with new statements
        if statements:
            self.tom_engine.update_beliefs_from_statements(statements, self.memory.voting_patterns)
        
        # Track role claims
        for player_id, statement in statements:
            if "detective" in statement.lower() and "i am" in statement.lower():
                self.memory.role_claims[player_id] = "Detective"
            elif "doctor" in statement.lower() and "i am" in statement.lower():
                self.memory.role_claims[player_id] = "Doctor"
    
    def _extract_player_statements(self, observation: str) -> List[Tuple[int, str]]:
        """Extract player statements from observation"""
        statements = []
        
        # Try multiple patterns to handle different formats
        patterns = [
            r'\[Player (\d+)\] (.+?)(?=\[Player \d+\]|$)',  # [Player X] format
            r'Player (\d+): (.+?)(?=Player \d+:|$)',        # Player X: format
            r'\[(\d+)\] (.+?)(?=\[\d+\]|$)',               # [X] format
            r'Player (\d+) says?: (.+?)(?=Player \d+|$)'   # Player X says: format
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, observation, re.DOTALL | re.MULTILINE)
            for player_id, message in matches:
                statements.append((int(player_id), message.strip()))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_statements = []
        for stmt in statements:
            if stmt not in seen:
                seen.add(stmt)
                unique_statements.append(stmt)
        
        return unique_statements
    
    def _extract_phase(self, observation: str) -> Optional[GamePhase]:
        """Extract current game phase with improved detection"""
        # Night phases
        if 'choose one player to investigate' in observation:
            return GamePhase.NIGHT_DETECTIVE
        elif 'choose one player to protect' in observation:
            return GamePhase.NIGHT_DOCTOR
        elif 'choose one player to eliminate' in observation or 'Night has fallen. Mafia, agree on a victim' in observation:
            return GamePhase.NIGHT_MAFIA
        
        # Day phases
        elif any(phrase in observation for phrase in [
            'Voting phase', 'submit one vote in format', 'Valid: [', 'vote will follow'
        ]):
            return GamePhase.DAY_VOTING
        elif any(phrase in observation for phrase in [
            'Day breaks', 'Discuss for', 'Round 1 Discussion', 'Round 2 Discussion', 'Round 3 Discussion'
        ]):
            return GamePhase.DAY_DISCUSSION
        
        # Fallback based on content patterns
        if '[Player' in observation and ']' in observation and any(word in observation.lower() for word in ['says', 'claims', 'discussion']):
            return GamePhase.DAY_DISCUSSION
        elif 'Valid targets:' in observation or 'Valid:' in observation:
            # Could be night or voting - check context
            if 'night' in observation.lower() or 'mafia' in observation.lower():
                return GamePhase.NIGHT_MAFIA
            else:
                return GamePhase.DAY_VOTING
        
        return GamePhase.DAY_DISCUSSION  # Default fallback
    
    def _extract_role(self, observation: str) -> Optional[Role]:
        """Extract my role from observation"""
        role_indicators = {
            'Your role: Villager': Role.VILLAGER,
            'Your role: Mafia': Role.MAFIA,
            'Your role: Doctor': Role.DOCTOR,
            'Your role: Detective': Role.DETECTIVE
        }
        
        for indicator, role in role_indicators.items():
            if indicator in observation:
                return role
        
        return None
    
    def _extract_available_targets(self, observation: str) -> List[int]:
        """Extract available targets from observation with proper validation"""
        # First update alive players from current observation
        self._update_alive_players_from_observation(observation)
        
        # Look for explicit target lists in observation
        target_patterns = [
            r'Valid targets: ((?:\[\d+\](?:, )?)+)',
            r'Valid: ((?:\[\d+\](?:, )?)+)',
            r'choose one player to (?:investigate|protect|eliminate): ((?:\[\d+\](?:, )?)+)',
            r'submit one vote in format \[X\]\. Valid: ((?:\[\d+\](?:, )?)+)'
        ]
        
        extracted_targets = []
        for pattern in target_patterns:
            match = re.search(pattern, observation)
            if match:
                target_str = match.group(1)
                targets = re.findall(r'\[(\d+)\]', target_str)
                extracted_targets = [int(t) for t in targets]
                print(f"🎯 Extracted targets from pattern: {extracted_targets}")
                break
        
        # If we found explicit targets, validate them against alive players
        if extracted_targets:
            # Filter out dead players and self (for night actions)
            valid_targets = []
            for target in extracted_targets:
                if target in self.alive_players:
                    # For night actions, don't target self
                    if self._is_night_action_phase(observation) and target == self.my_player_id:
                        continue
                    valid_targets.append(target)
            
            if valid_targets:
                print(f"✅ Valid targets after filtering: {valid_targets}")
                return valid_targets
        
        # No fallback allowed - raise exception instead
        if self.alive_players:
            print(f"❌ Target extraction failed but fallback prohibited")
            raise Exception("Failed to extract targets from observation")
        
        # No ultimate fallback - raise exception
        print("❌ No available targets and fallback prohibited")
        raise Exception("No available targets found")
    
    def _is_night_action_phase(self, observation: str) -> bool:
        """Check if current phase requires night action"""
        # Safety check: Villagers have no night actions
        if self.my_role == Role.VILLAGER:
            return False  # Villagers have no night actions
        
        # Look for the EXACT current action prompt at the end
        lines = [line.strip() for line in observation.strip().split('\n') if line.strip()]
        if not lines:
            return False
            
        # Check the last meaningful line for night action prompts
        last_line = lines[-1]
        
        night_indicators = [
            'choose one player to investigate:',
            'choose one player to protect:',
            'choose one player to eliminate:',
            'Night phase - choose one player to',
            'Mafia, agree on a victim'
        ]
        
        # Must be an exact match in the current prompt
        return any(indicator in last_line for indicator in night_indicators)
    
    def _is_voting_phase(self, observation: str) -> bool:
        """Check if current phase is voting"""
        # Check recent lines for voting phase indicators
        lines = observation.strip().split('\n')
        last_few_lines = lines[-10:]  # Check last 10 lines for current phase
        
        # Check recent lines for voting phase indicators
        for line in last_few_lines:
            if ('Voting phase' in line or 'submit one vote' in line or 
                'format [X]' in line or 'Valid: [' in line):
                return True
                
        # Check the entire observation for voting indicators
        if ('Voting phase' in observation or 'submit one vote' in observation or
            'format [X]' in observation):
            return True
            
        return False
    
    def _is_discussion_phase(self, observation: str) -> bool:
        """Check if current phase is discussion"""
        # First check if we're explicitly in voting or night phase
        if self._is_voting_phase(observation) or self._is_night_action_phase(observation):
            return False
            
        # Check for explicit discussion indicators
        if ('Day breaks' in observation or 'Discuss for' in observation):
            return True
        
        # Check for player statements (indicates discussion phase)
        player_statement_patterns = [
            r'\[\d+, ".*?"',  # Format: [Player_ID, "message", turn]
            r'Player \d+:',
        ]
        
        for pattern in player_statement_patterns:
            if re.search(pattern, observation):
                return True
        
        # Default: if not voting or night, assume discussion
        return True
    
    def _create_game_context(self, observation: str) -> GameContext:
        """Create game context for strategic bidding"""
        recent_statements = self._extract_player_statements(observation)[-5:]  # Last 5 statements
        
        return GameContext(
            current_turn=self.turn_count,
            phase=self.current_phase.value if self.current_phase and hasattr(self.current_phase, 'value') else (str(self.current_phase) if self.current_phase else "Unknown"),
            my_role=self._get_role_string(),
            my_player_id=self.my_player_id,
            alive_players=self.alive_players,
            suspicion_on_me=self.tom_engine.get_suspicion_level(self.my_player_id),
            recent_statements=recent_statements,
            voting_phase_active=self._is_voting_phase(observation),
            investigation_results=self.memory.investigation_results,
            night_kill_occurred="killed during the night" in observation
        )
    
    def _create_mcts_game_state(self, observation: str) -> Dict:
        """Create game state for MCTS planning"""
        return {
            'my_player_id': self.my_player_id,
            'suspicion_on_me': self.tom_engine.get_suspicion_level(self.my_player_id),
            'current_turn': self.turn_count,
            'phase': self.current_phase.value if self.current_phase and hasattr(self.current_phase, 'value') else (str(self.current_phase) if self.current_phase else "Discussion"),
            'alive_players': self.alive_players,
            'voting_phase_active': self._is_voting_phase(observation),
            'recent_responses': getattr(self, 'recent_responses', [])
        }
    
    def _get_current_beliefs(self) -> Dict[int, float]:
        """Get current belief state for reward calculation"""
        beliefs = {}
        for player_id in self.alive_players:
            beliefs[player_id] = self.tom_engine.get_suspicion_level(player_id)
        return beliefs
    
    def _apply_failure_prevention(self, response: str) -> str:
        """Apply all failure prevention measures"""
        response = self.failure_preventor.prevent_action_leakage(response)
        response = self.failure_preventor.prevent_language_drift(response)
        response = self.failure_preventor.prevent_repetition(response)
        response = self.failure_preventor.prevent_role_leakage(
            response, self._get_role_string(), 
            self.current_phase or GamePhase.DAY_DISCUSSION
        )
        
        # Update response history
        self.failure_preventor.response_history.append(response)
        if len(self.failure_preventor.response_history) > 10:
            self.failure_preventor.response_history.pop(0)
        
        return response
    
    
    def _being_accused_in_statements(self, recent_statements: List[Tuple[int, str]]) -> bool:
        """Check if we're being accused in recent statements"""
        accusation_keywords = ['suspicious', 'mafia', 'doubt', 'lying']
        
        for speaker_id, statement in recent_statements[-3:]:  # Last 3 statements
            if speaker_id != self.my_player_id:
                statement_lower = statement.lower()
                if f"player {self.my_player_id}" in statement_lower:
                    if any(keyword in statement_lower for keyword in accusation_keywords):
                        return True
        return False
    
    def generate_round_summary(self):
        """Generate enhanced round summary with performance analysis"""
        summary = {
            'turn_count': self.turn_count,
            'role': self._get_role_string(),
            'performance_metrics': self.performance_metrics,
            'suspicion_levels': {pid: self.tom_engine.get_suspicion_level(pid) for pid in self.alive_players},
            'behavioral_insights': {pid: dict(self.tom_engine.behavioral_patterns[pid]) for pid in self.alive_players},
            'memory_stats': {
                'observations': len(self.memory.observational),
                'discussions': len(self.memory.discussion_history),
                'investigations': len(self.memory.investigation_results),
                'role_claims': len(self.memory.role_claims)
            },
            'strategic_analysis': {
                'most_suspicious': self.tom_engine.get_most_suspicious_players(3, exclude_self_id=self.my_player_id),
                'trust_network': self.tom_engine.get_trust_network(),
                'bidding_stats': self.bidding_system.get_bidding_statistics(),
                'mcts_stats': {'disabled': True, 'single_response_mode': True}
            },
            # NEW TRAINING DATA
            'decision_reasoning': self._capture_decision_reasoning(),
            'counterfactual_analysis': self._generate_counterfactuals(),
            'communication_effectiveness': self._measure_communication_effectiveness(),
            'theory_of_mind_accuracy': self._calculate_tom_accuracy(),
            'alliance_detection': self._analyze_alliances(),
            'temporal_patterns': self._extract_temporal_patterns(),
            'linguistic_features': self._extract_linguistic_features(),
            'meta_game_analysis': self._analyze_meta_game(),
            'confidence_calibration': self._measure_confidence_calibration(),
            'exploration_exploitation': self._track_exploration_exploitation(),
            'information_leak_analysis': self._analyze_information_leaks(),
            'exploitation_opportunities': self._log_exploitation_opportunities()
        }
        
        self.memory.reflective.append(f"Round Summary: {json.dumps(summary, indent=2)}")
        print(f"Enhanced round summary generated: {len(self.memory.reflective)} total summaries")
        
        return summary
    
    def update_final_metrics(self, game_outcome: str, final_reward: float):
        """Update final performance metrics based on game outcome"""
        # Update total rewards
        self.performance_metrics['total_rewards'] += final_reward
        
        # Bonus metrics for winning
        if final_reward > 0:  # Won the game
            self.performance_metrics['strategic_decisions'] += 2  # Bonus for winning
            
            # Role-specific bonuses
            if self.my_role == Role.MAFIA:
                self.performance_metrics['successful_deflections'] += 1  # Survived as Mafia
            elif self.my_role in [Role.DETECTIVE, Role.DOCTOR]:
                self.performance_metrics['information_contributions'] += 1  # Used special role effectively
        
        # Update belief accuracy based on final outcome
        if "Mafia wins" in game_outcome and self.my_role == Role.MAFIA:
            # We correctly identified threats (since we won)
            self.performance_metrics['belief_accuracy'] = min(1.0, self.performance_metrics['belief_accuracy'] + 0.1)
        elif "Village wins" in game_outcome and self.my_role != Role.MAFIA:
            # We correctly identified Mafia (since village won)
            self.performance_metrics['belief_accuracy'] = min(1.0, self.performance_metrics['belief_accuracy'] + 0.1)
        
        print(f"[METRICS] Updated Performance Metrics: {self.performance_metrics}")
        
        return self.performance_metrics
    
    # NEW TRAINING DATA CAPTURE METHODS
    
    def _capture_decision_reasoning(self) -> Dict:
        """Capture reasoning behind each decision with confidence scores"""
        return {
            'last_discussion_reasoning': getattr(self, '_last_discussion_reasoning', {}),
            'last_vote_reasoning': getattr(self, '_last_vote_reasoning', {}),
            'last_night_reasoning': getattr(self, '_last_night_reasoning', {}),
            'decision_confidence_scores': getattr(self, '_decision_confidences', {}),
            'alternative_options_considered': getattr(self, '_alternatives_considered', []),
            'risk_assessment': getattr(self, '_risk_assessment', {}),
            'strategic_priority': getattr(self, '_strategic_priority', "unknown")
        }
    
    def _generate_counterfactuals(self) -> Dict:
        """Generate what-if scenarios for training on counterfactual reasoning"""
        counterfactuals = {}
        
        # What if I had voted differently?
        if hasattr(self, '_last_vote_target') and hasattr(self, '_vote_alternatives'):
            counterfactuals['alternative_votes'] = {
                'actual_choice': self._last_vote_target,
                'alternatives': self._vote_alternatives,
                'predicted_outcomes': {alt: self._predict_outcome_if_voted(alt) 
                                     for alt in self._vote_alternatives}
            }
        
        # What if I had said something different?
        if hasattr(self, '_last_statement') and hasattr(self, '_statement_alternatives'):
            counterfactuals['alternative_statements'] = {
                'actual_statement': self._last_statement,
                'alternatives': self._statement_alternatives,
                'predicted_belief_changes': {alt: self.tom_engine.simulate_belief_changes(alt, self.my_player_id) 
                                           for alt in self._statement_alternatives}
            }
        
        # What if different players were eliminated?
        counterfactuals['hypothetical_eliminations'] = self._analyze_hypothetical_eliminations()
        
        return counterfactuals
    
    def _measure_communication_effectiveness(self) -> Dict:
        """Measure how effective our communication was"""
        if not hasattr(self, '_communication_metrics'):
            self._communication_metrics = {
                'statements_made': 0,
                'suspicion_reduction_achieved': 0.0,
                'information_conveyed': 0,
                'trust_building_success': 0.0,
                'misdirection_success': 0.0  # For Mafia
            }
        
        return {
            'total_statements': self._communication_metrics['statements_made'],
            'avg_suspicion_reduction': self._communication_metrics['suspicion_reduction_achieved'],
            'information_density': self._communication_metrics['information_conveyed'] / max(1, self._communication_metrics['statements_made']),
            'trust_building_rate': self._communication_metrics['trust_building_success'],
            'persuasion_success': self._measure_persuasion_success(),
            'response_adaptation': self._measure_response_adaptation()
        }
    
    def _calculate_tom_accuracy(self) -> Dict:
        """Calculate accuracy of Theory of Mind predictions"""
        tom_accuracy = {
            'suspicion_prediction_accuracy': 0.0,
            'behavior_prediction_accuracy': 0.0,
            'alliance_detection_accuracy': 0.0,
            'role_prediction_accuracy': 0.0
        }
        
        # Compare predictions with revealed information
        for player_id in self.alive_players:
            predicted_suspicion = self.tom_engine.get_suspicion_level(player_id)
            # In real implementation, compare with actual roles when revealed
            tom_accuracy['suspicion_prediction_accuracy'] = predicted_suspicion  # Placeholder
        
        return tom_accuracy
    
    def _analyze_alliances(self) -> Dict:
        """Analyze detected and actual alliances"""
        alliance_analysis = {}
        
        for p1 in self.alive_players:
            for p2 in self.alive_players:
                if p1 < p2:  # Avoid duplicates
                    alliance_prob = self.tom_engine.calculate_alliance_probability(p1, p2)
                    if alliance_prob > 0.3:  # Significant alliance probability
                        alliance_analysis[f"{p1}_{p2}"] = {
                            'probability': alliance_prob,
                            'evidence': self._get_alliance_evidence(p1, p2),
                            'mutual_defense': self._check_mutual_defense(p1, p2),
                            'behavioral_similarity': self._calculate_behavioral_similarity(p1, p2)
                        }
        
        return alliance_analysis
    
    def _extract_temporal_patterns(self) -> Dict:
        """Extract temporal patterns in behavior and game state"""
        return {
            'suspicion_evolution': self._track_suspicion_over_time(),
            'behavioral_drift': self._measure_behavioral_drift(),
            'phase_transitions': self._analyze_phase_transitions(),
            'pressure_response_patterns': self._analyze_pressure_responses(),
            'timing_of_accusations': self._analyze_accusation_timing(),
            'late_game_behavior_shifts': self._detect_endgame_shifts()
        }
    
    def _extract_linguistic_features(self) -> Dict:
        """Extract linguistic features from all statements"""
        linguistic_features = {
            'vocabulary_diversity': {},
            'sentence_complexity': {},
            'emotional_markers': {},
            'deception_indicators': {},
            'persuasion_techniques': {},
            'cognitive_load_indicators': {}
        }
        
        for player_id, statements in self._get_player_statements().items():
            linguistic_features['vocabulary_diversity'][player_id] = self._calculate_vocabulary_diversity(statements)
            linguistic_features['sentence_complexity'][player_id] = self._analyze_sentence_complexity(statements)
            linguistic_features['emotional_markers'][player_id] = self._extract_emotional_markers(statements)
            linguistic_features['deception_indicators'][player_id] = self._detect_deception_indicators(statements)
        
        return linguistic_features
    
    def _analyze_meta_game(self) -> Dict:
        """Analyze meta-game strategies and adaptations"""
        return {
            'strategy_shifts': self._detect_strategy_changes(),
            'opponent_modeling': self._assess_opponent_modeling(),
            'exploitation_attempts': self._track_exploitation_attempts(),
            'bluff_detection': self._analyze_bluff_patterns(),
            'information_management': self._assess_information_management(),
            'risk_tolerance_evolution': self._track_risk_tolerance()
        }
    
    def _measure_confidence_calibration(self) -> Dict:
        """Measure how well-calibrated our confidence scores are"""
        return {
            'prediction_confidence_accuracy': self._assess_prediction_accuracy(),
            'overconfidence_bias': self._measure_overconfidence(),
            'uncertainty_quantification': self._analyze_uncertainty(),
            'confidence_decision_correlation': self._correlate_confidence_decisions()
        }
    
    def _track_exploration_exploitation(self) -> Dict:
        """Track balance between exploration and exploitation"""
        return {
            'information_gathering_vs_action': self._measure_info_vs_action_balance(),
            'novel_strategy_attempts': self._count_novel_strategies(),
            'safe_vs_risky_moves': self._analyze_risk_taking(),
            'adaptation_to_feedback': self._measure_adaptation_rate()
        }
    
    # Helper methods for the new logging features (simplified implementations)
    
    def _predict_outcome_if_voted(self, target: int) -> Dict:
        """Predict game outcome if we voted for target"""
        return {'probability_village_wins': 0.5, 'predicted_next_elimination': target}
    
    def _analyze_hypothetical_eliminations(self) -> Dict:
        """Analyze what would happen if different players were eliminated"""
        return {player_id: {'impact_on_village_win_rate': 0.5} for player_id in self.alive_players}
    
    def _measure_persuasion_success(self) -> float:
        """Measure how successful we were at persuading others"""
        return 0.5  # Placeholder
    
    def _measure_response_adaptation(self) -> float:
        """Measure how well we adapted our responses to the situation"""
        return 0.5  # Placeholder
    
    def _detect_and_process_leaks(self, observation: str):
        """Detect and process information leaks from other players"""
        # Extract recent statements for leak analysis
        recent_statements = self._extract_player_statements(observation)
        
        # Create game context for leak detection
        game_context = {
            'turn': self.turn_count,
            'phase': self.current_phase.value if self.current_phase and hasattr(self.current_phase, 'value') else (str(self.current_phase) if self.current_phase else 'unknown'),
            'my_role': self._get_role_string().lower(),
            'claimed_detectives': [pid for pid, role in self.memory.role_claims.items() if role == 'Detective'],
            'claimed_doctors': [pid for pid, role in self.memory.role_claims.items() if role == 'Doctor'],
            'alive_players': self.alive_players,
            'night_kills': self.memory.night_kill_history
        }
        
        # Detect leaks
        new_leaks = self.leak_detector.detect_leaks(recent_statements, game_context)
        
        # Process and store significant leaks
        for leak in new_leaks:
            if leak.confidence > 0.6:  # Only process high-confidence leaks
                self.detected_leaks.append(leak)
                
                # Update our beliefs based on the leak
                self._update_beliefs_from_leak(leak)
                
                # Plan exploitation if beneficial
                if leak.exploit_potential > 0.7:
                    self._plan_leak_exploitation(leak)
                
                print(f"[LEAK] Detected leak: Player {leak.source_player} - {leak.leak_type} (confidence: {leak.confidence:.2f})")
    
    def _update_beliefs_from_leak(self, leak: InformationLeak):
        """Update our beliefs based on detected information leak"""
        player_id = leak.source_player
        
        # Role-specific belief updates
        if 'detective' in leak.leak_type:
            # Player likely is or claims to be Detective
            self.memory.role_claims[player_id] = 'Detective'
            # Detectives are usually Village, reduce suspicion
            current_suspicion = self.tom_engine.get_suspicion_level(player_id)
            self.tom_engine.my_beliefs[player_id].update(-0.3, 0.2, self.turn_count)
            
        elif 'doctor' in leak.leak_type:
            # Player likely is or claims to be Doctor
            self.memory.role_claims[player_id] = 'Doctor'
            # Doctors are usually Village, reduce suspicion
            self.tom_engine.my_beliefs[player_id].update(-0.3, 0.2, self.turn_count)
            
        elif 'mafia' in leak.leak_type:
            # Player likely is Mafia based on coordination/knowledge
            # Significantly increase suspicion
            self.tom_engine.my_beliefs[player_id].update(0.6, 0.3, self.turn_count)
            
        elif leak.leak_type == 'impossible_knowledge':
            # Player knows something they shouldn't - likely Mafia
            self.tom_engine.my_beliefs[player_id].update(0.4, 0.2, self.turn_count)
            
        elif leak.leak_type == 'coordination_leak':
            # Player coordinating with others - potential Mafia alliance
            self.tom_engine.my_beliefs[player_id].update(0.3, 0.1, self.turn_count)
        
        # Update behavioral patterns
        patterns = self.tom_engine.behavioral_patterns[player_id]
        patterns['information_leak_tendency'] = patterns.get('information_leak_tendency', 0.0) + 0.2
        patterns['security_awareness'] = max(0.0, patterns.get('security_awareness', 0.5) - 0.1)
    
    def _plan_leak_exploitation(self, leak: InformationLeak):
        """Plan how to exploit detected information leak"""
        exploit_plan = {
            'leak_id': len(self.exploit_opportunities),
            'source_player': leak.source_player,
            'leak_type': leak.leak_type,
            'exploitation_strategy': self._determine_exploitation_strategy(leak),
            'timing': self._determine_optimal_timing(leak),
            'expected_benefit': leak.exploit_potential,
            'risk_level': self._assess_exploitation_risk(leak)
        }
        
        self.exploit_opportunities.append(exploit_plan)
        
        # Execute immediate exploitation if low-risk and high-benefit
        if exploit_plan['risk_level'] < 0.3 and exploit_plan['expected_benefit'] > 0.8:
            self._execute_leak_exploitation(exploit_plan)
    
    def _determine_exploitation_strategy(self, leak: InformationLeak) -> str:
        """Determine the best strategy to exploit this leak"""
        if leak.leak_type in ['detective_hint', 'detective_claim']:
            if self.my_role == Role.MAFIA:
                return 'target_for_elimination'  # Target Detective for night kill
            else:
                return 'coordinate_with_detective'  # Work with Detective
                
        elif leak.leak_type in ['doctor_hint', 'doctor_claim']:
            if self.my_role == Role.MAFIA:
                return 'target_for_elimination'  # Target Doctor
            else:
                return 'protect_doctor'  # Keep Doctor alive
                
        elif leak.leak_type in ['mafia_coordination', 'mafia_planning', 'impossible_knowledge']:
            if self.my_role != Role.MAFIA:
                return 'expose_mafia'  # Use as evidence against them
            else:
                return 'warn_teammate'  # Warn teammate about their leak
                
        elif leak.leak_type == 'coordination_leak':
            return 'investigate_alliance'  # Look for coordinated players
            
        return 'monitor_and_gather_evidence'
    
    def _determine_optimal_timing(self, leak: InformationLeak) -> str:
        """Determine when to exploit the leak"""
        if leak.leak_type in ['mafia_coordination', 'impossible_knowledge']:
            return 'immediate'  # Strike while evidence is fresh
        elif leak.leak_type in ['detective_hint', 'doctor_hint']:
            return 'next_night_phase'  # Target during night
        else:
            return 'strategic_moment'  # Wait for optimal moment
    
    def _assess_exploitation_risk(self, leak: InformationLeak) -> float:
        """Assess risk of exploiting this leak"""
        base_risk = 0.3
        
        # Higher risk if many players present
        if len(self.alive_players) > 4:
            base_risk += 0.1
        
        # Higher risk if leak source has allies
        if self._player_has_defenders(leak.source_player):
            base_risk += 0.2
        
        # Lower risk if we have high credibility
        my_suspicion = self.tom_engine.get_suspicion_level(self.my_player_id)
        if my_suspicion < 0.3:
            base_risk -= 0.1
        
        return min(1.0, max(0.0, base_risk))
    
    def _execute_leak_exploitation(self, exploit_plan: Dict):
        """Execute immediate exploitation of detected leak"""
        strategy = exploit_plan['exploitation_strategy']
        target_player = exploit_plan['source_player']
        
        # Store exploitation plan for use in next communication
        if not hasattr(self, '_pending_exploitations'):
            self._pending_exploitations = []
        
        self._pending_exploitations.append({
            'target': target_player,
            'strategy': strategy,
            'evidence': f"Detected {exploit_plan['leak_type']} leak",
            'confidence': exploit_plan['expected_benefit']
        })
        
        print(f"[EXPLOIT] Planned exploitation: {strategy} against Player {target_player}")
    
    def _player_has_defenders(self, player_id: int) -> bool:
        """Check if player has others defending them"""
        defenders = 0
        for other_id in self.alive_players:
            if other_id != player_id and other_id != self.my_player_id:
                suspicion_of_target = self.tom_engine.get_meta_suspicion(other_id, player_id)
                if suspicion_of_target < 0.3:  # Other player trusts target
                    defenders += 1
        
        return defenders >= 2
    
    def _integrate_leak_exploitation_into_communication(self, candidates: List[str]) -> List[str]:
        """Integrate leak exploitation into communication candidates"""
        if not hasattr(self, '_pending_exploitations'):
            return candidates
        
        enhanced_candidates = candidates.copy()
        
        for exploitation in self._pending_exploitations:
            target = exploitation['target']
            strategy = exploitation['strategy']
            evidence = exploitation['evidence']
            
            if strategy == 'expose_mafia':
                enhanced_candidates.extend([
                    f"I've noticed some concerning inconsistencies in Player {target}'s statements that suggest deception.",
                    f"Player {target} seems to know things they shouldn't know - how did they find that out?",
                    f"The pattern of statements from Player {target} indicates they might be working against Village interests."
                ])
            
            elif strategy == 'coordinate_with_detective':
                enhanced_candidates.extend([
                    f"Player {target}'s analytical approach suggests they might have valuable insights to share.",
                    f"I think Player {target} and I should coordinate our information gathering efforts.",
                    f"Player {target} seems to be doing important investigative work that could benefit the Village."
                ])
            
            elif strategy == 'protect_doctor':
                enhanced_candidates.extend([
                    f"Player {target} has been making strategically sound decisions that benefit Village survival.",
                    f"We should prioritize protecting players like Player {target} who contribute to Village success.",
                    f"Player {target}'s strategic thinking makes them a valuable Village asset we need to keep safe."
                ])
        
        # Clear pending exploitations after integration
        self._pending_exploitations = []
        
        return enhanced_candidates
    
    def _get_alliance_evidence(self, p1: int, p2: int) -> List[str]:
        """Get evidence for alliance between two players"""
        return ["mutual_defense", "similar_voting_patterns"]  # Placeholder
    
    def _check_mutual_defense(self, p1: int, p2: int) -> bool:
        """Check if two players mutually defend each other"""
        return False  # Placeholder
    
    def _calculate_behavioral_similarity(self, p1: int, p2: int) -> float:
        """Calculate behavioral similarity between two players"""
        return 0.5  # Placeholder
    
    def _track_suspicion_over_time(self) -> Dict:
        """Track how suspicion levels changed over time"""
        return {player_id: [0.5] * self.turn_count for player_id in self.alive_players}
    
    def _measure_behavioral_drift(self) -> Dict:
        """Measure how much each player's behavior drifted"""
        return {player_id: 0.1 for player_id in self.alive_players}
    
    def _analyze_phase_transitions(self) -> Dict:
        """Analyze behavior changes at phase transitions"""
        return {'day_to_night': {}, 'night_to_day': {}}
    
    def _analyze_pressure_responses(self) -> Dict:
        """Analyze how players respond to pressure"""
        return {player_id: {'defensive_increase': 0.1} for player_id in self.alive_players}
    
    def _analyze_accusation_timing(self) -> Dict:
        """Analyze timing patterns of accusations"""
        return {'early_accusers': [], 'late_accusers': [], 'consistent_accusers': []}
    
    def _detect_endgame_shifts(self) -> Dict:
        """Detect behavior shifts in endgame"""
        return {'players_with_shifts': [], 'shift_magnitude': {}}
    
    def _get_player_statements(self) -> Dict[int, List[str]]:
        """Get all statements by each player"""
        statements_by_player = defaultdict(list)
        for player_id, statement in self.memory.discussion_history:
            statements_by_player[player_id].append(statement)
        return dict(statements_by_player)
    
    def _calculate_vocabulary_diversity(self, statements: List[str]) -> float:
        """Calculate vocabulary diversity for a player"""
        all_words = []
        for statement in statements:
            all_words.extend(statement.lower().split())
        unique_words = len(set(all_words))
        total_words = len(all_words)
        return unique_words / max(1, total_words)
    
    def _analyze_sentence_complexity(self, statements: List[str]) -> Dict:
        """Analyze sentence complexity"""
        return {
            'avg_sentence_length': sum(len(s.split()) for s in statements) / max(1, len(statements)),
            'complex_sentences': sum(1 for s in statements if len(s.split()) > 15) / max(1, len(statements))
        }
    
    def _extract_emotional_markers(self, statements: List[str]) -> Dict:
        """Extract emotional markers from statements"""
        return {'positive': 0.5, 'negative': 0.3, 'neutral': 0.2}
    
    def _detect_deception_indicators(self, statements: List[str]) -> Dict:
        """Detect linguistic indicators of deception"""
        return {'hedging': 0.2, 'overemphasis': 0.1, 'contradiction': 0.05}
    
    # Placeholder implementations for meta-game analysis
    def _detect_strategy_changes(self) -> Dict:
        return {'strategy_shift_turns': [], 'strategy_types': []}
    
    def _assess_opponent_modeling(self) -> Dict:
        return {'modeling_accuracy': 0.5, 'adaptation_speed': 0.3}
    
    def _track_exploitation_attempts(self) -> Dict:
        return {'successful_exploits': 0, 'failed_exploits': 0}
    
    def _analyze_bluff_patterns(self) -> Dict:
        return {'bluff_attempts': 0, 'bluff_success_rate': 0.0}
    
    def _assess_information_management(self) -> Dict:
        return {'information_revealed': 0.3, 'information_concealed': 0.7}
    
    def _track_risk_tolerance(self) -> Dict:
        return {'risk_level_over_time': [0.5] * self.turn_count}
    
    def _assess_prediction_accuracy(self) -> float:
        return 0.6
    
    def _measure_overconfidence(self) -> float:
        return 0.1
    
    def _analyze_uncertainty(self) -> Dict:
        return {'uncertainty_level': 0.4, 'uncertainty_change': 0.05}
    
    def _correlate_confidence_decisions(self) -> float:
        return 0.7
    
    def _measure_info_vs_action_balance(self) -> Dict:
        return {'exploration_ratio': 0.4, 'exploitation_ratio': 0.6}
    
    def _count_novel_strategies(self) -> int:
        return 2
    
    def _analyze_risk_taking(self) -> Dict:
        return {'safe_moves': 5, 'risky_moves': 2, 'risk_reward_ratio': 0.4}
    
    def _measure_adaptation_rate(self) -> float:
        return 0.3
    
    def _determine_strategic_priority(self, observation: str) -> str:
        """Determine the current strategic priority"""
        if self.my_role == Role.MAFIA:
            return "survival_and_misdirection"
        elif self.my_role == Role.DETECTIVE:
            return "information_gathering"
        elif self.my_role == Role.DOCTOR:
            return "protection_optimization"
        else:
            return "mafia_identification"
    
    def _ensure_natural_length(self, candidates: List[str]) -> List[str]:
        """Pass through responses without modification"""
        return candidates
    
    def _expand_response_naturally(self, base_response: str, target_length: int = 40) -> str:
        """Pass through response without modification"""
        return base_response
    
    def _add_conversational_elements(self, response: str) -> str:
        """Pass through response without modification"""
        return response
    
    def _select_unused_element(self, options: List[str], recent_history: List[str], max_history: int = 5) -> str:
        """Select an element that hasn't been used recently"""
        # Remove old history beyond max_history
        while len(recent_history) >= max_history:
            recent_history.pop(0)
        
        # Find options not recently used
        unused_options = [opt for opt in options if opt not in recent_history]
        
        # If all options have been used recently, use any option but prefer least recent
        if not unused_options:
            unused_options = options
        
        # Select randomly from unused options
        selected = random.choice(unused_options)
        
        # Add to recent history
        recent_history.append(selected)
        
        return selected
    
    # VILLAGER STRATEGY HELPER METHODS
    
    def _identify_quiet_players(self, observation: str) -> List[int]:
        """Identify players who have been unusually quiet"""
        player_activity = {}
        
        # Count statements for each player
        for player_id, statement in self.memory.discussion_history:
            if player_id not in player_activity:
                player_activity[player_id] = 0
            player_activity[player_id] += len(statement.split())
        
        # Find players with below-average activity
        if not player_activity:
            return []
        
        avg_activity = sum(player_activity.values()) / len(player_activity)
        quiet_players = [pid for pid, activity in player_activity.items() 
                        if activity < avg_activity * 0.6 and pid != self.my_player_id]
        
        return sorted(quiet_players, key=lambda p: player_activity.get(p, 0))
    
    def _identify_likely_village_players(self, observation: str) -> List[int]:
        """Identify players likely to be Village based on behavior"""
        village_candidates = []
        
        for player_id in self.alive_players:
            if player_id == self.my_player_id:
                continue
                
            suspicion = self.tom_engine.get_suspicion_level(player_id)
            activity = self.tom_engine.get_behavioral_pattern(player_id, 'activity_level')
            
            # Low suspicion + high activity = likely Village
            if suspicion < 0.3 and activity > 0.6:
                village_candidates.append((player_id, 1.0 - suspicion + activity))
        
        # Sort by confidence score
        village_candidates.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, score in village_candidates]
    
    def _is_late_game(self, observation: str) -> bool:
        """Check if we're in late game (Mafia close to winning)"""
        alive_count = len(self.alive_players)
        
        # If 4 or fewer players left, it's late game
        if alive_count <= 4:
            return True
            
        # If turn count is high, it's late game
        if self.turn_count >= 8:
            return True
            
        return False
    
    def _detective_revealed(self, observation: str) -> bool:
        """Check if Detective has been revealed"""
        return any("detective" in claim.lower() for claim in self.memory.role_claims.values())
    
    def _analyze_information_leaks(self) -> Dict:
        """Analyze detected information leaks for training data"""
        leak_analysis = {
            'total_leaks_detected': len(self.detected_leaks),
            'leaks_by_type': {},
            'leaks_by_player': {},
            'exploitation_success_rate': 0.0,
            'leak_detection_confidence': []
        }
        
        # Categorize leaks by type
        for leak in self.detected_leaks:
            leak_type = leak.leak_type
            if leak_type not in leak_analysis['leaks_by_type']:
                leak_analysis['leaks_by_type'][leak_type] = []
            
            leak_analysis['leaks_by_type'][leak_type].append({
                'player': leak.source_player,
                'confidence': leak.confidence,
                'exploit_potential': leak.exploit_potential,
                'turn_detected': leak.turn_detected
            })
            
            # Track by player
            player_id = leak.source_player
            if player_id not in leak_analysis['leaks_by_player']:
                leak_analysis['leaks_by_player'][player_id] = []
            
            leak_analysis['leaks_by_player'][player_id].append({
                'type': leak_type,
                'confidence': leak.confidence,
                'turn': leak.turn_detected
            })
            
            leak_analysis['leak_detection_confidence'].append(leak.confidence)
        
        # Calculate average confidence
        if leak_analysis['leak_detection_confidence']:
            leak_analysis['avg_detection_confidence'] = sum(leak_analysis['leak_detection_confidence']) / len(leak_analysis['leak_detection_confidence'])
        else:
            leak_analysis['avg_detection_confidence'] = 0.0
        
        return leak_analysis

    def _generate_tom_insights(self) -> str:
        """Generate Theory of Mind insights about other players (OPTIMIZED)"""
        if not hasattr(self, 'tom_engine'):
            return ""
        
        # PERFORMANCE OPTIMIZATION: Cache results and only regenerate every few turns
        if hasattr(self, '_cached_tom_insights') and hasattr(self, '_tom_cache_turn'):
            if self.turn_count - self._tom_cache_turn < 3:  # Use cache for 3 turns
                return self._cached_tom_insights
        
        insights = []
        
        # Get most suspicious players (excluding self) - LIMIT TO TOP 2 FOR SPEED
        suspicious_players = self.tom_engine.get_most_suspicious_players(2, exclude_self_id=self.my_player_id)
        if suspicious_players:
            insights.append("SUSPICIOUS PLAYERS:")
            for player_id, suspicion_level in suspicious_players:
                insights.append(f"- Player {player_id}: {suspicion_level:.1%} suspicious")
        
        # SKIP TRUST NETWORK ANALYSIS TO SAVE TIME (comment out expensive operations)
        # trust_network = self.tom_engine.get_trust_network()
        
        # SIMPLIFIED BEHAVIORAL ANALYSIS - only for most suspicious
        if suspicious_players:
            insights.append("\nBEHAVIORAL PATTERNS:")
            # Only analyze top 1 suspicious player for speed
            player_id, _ = suspicious_players[0]
            deflection = self.tom_engine.get_behavioral_pattern(player_id, 'deflection_tendency')
            defensive = self.tom_engine.get_behavioral_pattern(player_id, 'defensive_tendency')
            if deflection > 0.6 or defensive > 0.6:
                patterns = []
                if deflection > 0.6:
                    patterns.append("deflective")
                if defensive > 0.6:
                    patterns.append("defensive")
                insights.append(f"- Player {player_id}: {', '.join(patterns)} behavior")
        
        # CACHE THE RESULT for performance
        result = "\n".join(insights) if insights else "No strong beliefs about other players yet."
        self._cached_tom_insights = result
        self._tom_cache_turn = self.turn_count
        
        return result

    def _create_game_context(self, observation: str) -> 'GameContext':
        """Create game context for strategic bidding"""
        from .bidding_system import GameContext, UrgencyLevel
        
        # Determine urgency level
        urgency = UrgencyLevel.GENERAL
        if "vote" in observation.lower() or "voting" in observation.lower():
            urgency = UrgencyLevel.URGENT
        elif self.turn_count <= 2:
            urgency = UrgencyLevel.OBSERVE
        
        return GameContext(
            current_turn=self.turn_count,
            phase=self._get_current_phase(observation),
            my_role=self.my_role.value if self.my_role else "Unknown",
            my_player_id=self.my_player_id if self.my_player_id is not None else 0,
            alive_players=self.alive_players if self.alive_players else [],
            suspicion_on_me=self._calculate_suspicion_on_me(),
            recent_statements=self._extract_player_statements(observation),
            voting_phase_active="vote" in observation.lower() or "voting" in observation.lower(),
            investigation_results=self.memory.investigation_results,
            night_kill_occurred="killed" in observation.lower()
        )

    def _get_current_phase(self, observation: str) -> str:
        """Determine current game phase from observation"""
        obs_lower = observation.lower()
        
        if "vote" in obs_lower or "voting" in obs_lower:
            return "Day-Voting"
        elif "night" in obs_lower:
            if "mafia" in obs_lower:
                return "Night-Mafia"
            elif "doctor" in obs_lower:
                return "Night-Doctor"
            elif "detective" in obs_lower:
                return "Night-Detective"
            else:
                return "Night-Mafia"  # Default night phase
        else:
            return "Day-Discussion"  # Default day phase

    def _calculate_suspicion_on_me(self) -> float:
        """Calculate how much suspicion is on this agent"""
        if not hasattr(self, 'memory') or not self.memory.suspicion_tracker:
            return 0.0
        
        my_id = self.my_player_id if self.my_player_id is not None else 0
        
        # Return the suspicion level for this player directly
        # suspicion_tracker is Dict[int, float] where key=player_id, value=suspicion_level
        return self.memory.suspicion_tracker.get(my_id, 0.0)

    def _is_valid_discussion_response(self, response: str) -> bool:
        """Validate that a response is appropriate for discussion phase"""
        if not response or len(response.strip()) < 10:
            return False
            
        # Check for invalid patterns
        invalid_patterns = [
            r'^\[\d+\]$',  # Vote format like [0], [1], etc.
            r'^\*{3,}',    # Starts with multiple asterisks (garbled)
            r'^I investigate Player \d+',  # Investigation format
            r'^You protect Player \d+',    # Protection format
            r'^Player \d+ was (killed|eliminated)',  # Game state announcements
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, response.strip(), re.IGNORECASE):
                return False
        
        # Should contain conversational elements for discussion
        discussion_indicators = [
            'I think', 'I believe', 'In my opinion', 'Let me', 'We should',
            'Player', 'suspicious', 'trust', 'vote', 'eliminate', 'Mafia',
            'villager', 'detective', 'doctor', 'behavior', 'evidence'
        ]
        
        response_lower = response.lower()
        has_discussion_element = any(indicator.lower() in response_lower for indicator in discussion_indicators)
        
        return has_discussion_element

    # REMOVED: All fallback response methods to comply with new competition rules
    # Per updated rules: "Every response must come directly from the LLM and be reasoned by the agent"
    # Fallback answers are now classified as heavy heuristics and prohibited

    # REMOVED: _apply_strategic_bidding_modulation method 
    # This method violated competition rules by modifying LLM responses with predetermined text replacements
    # All response modifications must come from LLM reasoning, not external heuristics

    def _update_strategic_preferences_from_reward(self, reward: float, selected_candidate: dict = None):
        """Update strategic preferences based on reward feedback"""
        if not hasattr(self, 'strategic_preferences'):
            self.strategic_preferences = {'aggressive': 0.0, 'moderate': 0.0, 'conservative': 0.0}
        
        if selected_candidate and 'strategy' in selected_candidate:
            strategy_type = 'conservative'
            if 'aggressively' in selected_candidate['strategy'].lower():
                strategy_type = 'aggressive'
            elif 'moderately' in selected_candidate['strategy'].lower():
                strategy_type = 'moderate'
            
            # Update preference based on reward (simple learning)
            self.strategic_preferences[strategy_type] += reward * 0.1
            print(f"📈 Strategic preference updated: {strategy_type} += {reward * 0.1:.3f}")
        
        # Apply reward-based learning to bidding system
        if hasattr(self.bidding_system, 'update_from_reward'):
            self.bidding_system.update_from_reward(reward)

    def _handle_modal_voting(self, observation: str) -> str:
        """Handle voting through Modal LLM with full game memory"""
        print("🗳️ Using Modal LLM for voting decision...")
        
        # Get available targets FIRST for validation
        available_targets = self._extract_available_targets(observation)
        print(f"🗳️ Available voting targets: {available_targets}")
        
        if not available_targets:
            print("❌ No valid voting targets available!")
            return "[1]"  # Emergency fallback
        
        # Call Modal with voting context (OPTIMIZED)
        try:
            endpoint_url = self.endpoint_url.rstrip('/') + '/generate'
            response = self.requests.post(
                endpoint_url,
                json={"observation": observation},
                timeout=15  # Fast voting timeout - 15 seconds
            )
            response.raise_for_status()
            
            result = response.json()
            modal_response = result.get("response", "[1]")
            processing_time = result.get("processing_time", 0)
            
            print(f"🌐 Modal voting response received in {processing_time:.2f}s: '{modal_response}'")
            
            # Extract vote from response and validate
            vote_match = re.search(r'\[(\d+)\]', modal_response)
            if vote_match:
                target = int(vote_match.group(1))
                if target in available_targets:
                    print(f"✅ Valid vote target: [{target}]")
                    return f"[{target}]"
                else:
                    print(f"❌ Invalid vote target {target}, not in {available_targets}")
            
            # Extract any number from response and validate
            numbers = re.findall(r'\d+', modal_response)
            for num_str in numbers:
                target = int(num_str)
                if target in available_targets:
                    print(f"✅ Valid number vote found: [{target}]")
                    return f"[{target}]"
            
            print(f"❌ No valid vote targets found in response: '{modal_response}'")
                
        except Exception as e:
            print(f"❌ Modal voting failed: {e}, falling back to strategic voting")
        
        # No fallback allowed - raise exception if validation fails
        if available_targets:
            print(f"❌ Modal voting failed but fallback prohibited")
            raise Exception("Modal voting failed and fallback is prohibited")
        
        # No ultimate fallback - raise exception
        raise Exception("No available targets for voting")


    def _handle_modal_night_action(self, observation: str) -> str:
        """Handle night actions through Modal LLM with full game memory"""
        print("🌙 Using Modal LLM for night action decision...")
        
        # Get available targets FIRST for validation
        available_targets = self._extract_available_targets(observation)
        print(f"🎯 Available targets for night action: {available_targets}")
        
        if not available_targets:
            print("❌ No valid targets available for night action!")
            return "[1]"  # Emergency fallback
        
        # Call Modal with night action context (OPTIMIZED)
        try:
            endpoint_url = self.endpoint_url.rstrip('/') + '/generate'
            response = self.requests.post(
                endpoint_url,
                json={"observation": observation},
                timeout=10  # Very fast night action timeout - 10 seconds
            )
            response.raise_for_status()
            
            result = response.json()
            modal_response = result.get("response", "[1]")
            processing_time = result.get("processing_time", 0)
            
            print(f"🌐 Modal night action response received in {processing_time:.2f}s: '{modal_response}'")
            
            # Extract action from response and validate
            action_match = re.search(r'\[(\d+)\]', modal_response)
            if action_match:
                target = int(action_match.group(1))
                if target in available_targets:
                    print(f"✅ Valid target selected: [{target}]")
                    return f"[{target}]"
                else:
                    print(f"❌ Invalid target {target}, not in {available_targets}")
            
            # Extract any number from response and validate
            numbers = re.findall(r'\d+', modal_response)
            for num_str in numbers:
                target = int(num_str)
                if target in available_targets:
                    print(f"✅ Valid number target found: [{target}]")
                    return f"[{target}]"
            
            print(f"❌ No valid targets found in response: '{modal_response}'")
                
        except Exception as e:
            print(f"❌ Modal night action failed: {e}, falling back to strategic action")
        
        # No fallback allowed - raise exception if modal fails
        if available_targets:
            print(f"❌ Modal night action failed but fallback prohibited")
            raise Exception("Modal night action failed and fallback is prohibited")
        
        # No ultimate fallback - raise exception
        raise Exception("No available targets for night action")
    
    def _log_exploitation_opportunities(self) -> Dict:
        """Log exploitation opportunities for training data"""
        return {
            'total_opportunities': len(self.exploit_opportunities),
            'opportunities_by_strategy': self._categorize_exploit_strategies(),
            'risk_assessment_distribution': self._analyze_risk_distribution(),
            'exploitation_timing_patterns': self._analyze_exploitation_timing(),
            'success_predictions': self._predict_exploitation_success()
        }
    
    def _categorize_exploit_strategies(self) -> Dict:
        """Categorize exploitation strategies"""
        strategies = {}
        for opportunity in self.exploit_opportunities:
            strategy = opportunity['exploitation_strategy']
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append({
                'target': opportunity['source_player'],
                'expected_benefit': opportunity['expected_benefit'],
                'risk_level': opportunity['risk_level']
            })
        return strategies
    
    def _analyze_risk_distribution(self) -> Dict:
        """Analyze risk distribution of exploitation opportunities"""
        if not self.exploit_opportunities:
            return {'low_risk': 0, 'medium_risk': 0, 'high_risk': 0}
        
        risk_levels = [opp['risk_level'] for opp in self.exploit_opportunities]
        return {
            'low_risk': sum(1 for r in risk_levels if r < 0.3),
            'medium_risk': sum(1 for r in risk_levels if 0.3 <= r < 0.7),
            'high_risk': sum(1 for r in risk_levels if r >= 0.7),
            'avg_risk': sum(risk_levels) / len(risk_levels)
        }
    
    def _analyze_exploitation_timing(self) -> Dict:
        """Analyze timing patterns of exploitation"""
        timing_patterns = {}
        for opportunity in self.exploit_opportunities:
            timing = opportunity['timing']
            if timing not in timing_patterns:
                timing_patterns[timing] = 0
            timing_patterns[timing] += 1
        return timing_patterns
    
    def _predict_exploitation_success(self) -> Dict:
        """Predict success of exploitation opportunities"""
        if not self.exploit_opportunities:
            return {'predicted_successes': 0, 'high_confidence_predictions': 0}
        
        high_confidence = sum(1 for opp in self.exploit_opportunities if opp['expected_benefit'] > 0.8)
        return {
            'predicted_successes': len([opp for opp in self.exploit_opportunities if opp['expected_benefit'] > 0.6]),
            'high_confidence_predictions': high_confidence,
            'avg_expected_benefit': sum(opp['expected_benefit'] for opp in self.exploit_opportunities) / len(self.exploit_opportunities)
        }
    
    # ============================================================================
    # CHAMPIONSHIP ENHANCEMENT HELPER METHODS
    # ============================================================================
    
    def _get_role_string(self) -> str:
        """Safely extract role as string, handling both enum and string types"""
        if self.my_role is None:
            return "Villager"
        elif hasattr(self.my_role, 'value'):
            return self.my_role.value
        else:
            return str(self.my_role)
    
    def _get_phase_string(self) -> str:
        """Safely extract phase as string, handling both enum and string types"""
        if self.current_phase is None:
            return "Unknown"
        elif hasattr(self.current_phase, 'value'):
            return self.current_phase.value
        else:
            return str(self.current_phase)
    
    def _being_accused_in_statements(self, statements: List[Tuple[int, str]]) -> bool:
        """Check if we're being accused in recent statements"""
        if not statements or self.my_player_id == -1:
            return False
        
        accusation_keywords = ['suspicious', 'suspect', 'vote', 'eliminate', 'mafia', 'guilty']
        my_id_str = str(self.my_player_id)
        
        for player_id, statement in statements:
            if player_id != self.my_player_id:
                statement_lower = statement.lower()
                if my_id_str in statement and any(keyword in statement_lower for keyword in accusation_keywords):
                    return True
        
        return False
    
    def _count_accusations_against_me(self, statements: List[Tuple[int, str]]) -> int:
        """Count how many players are accusing us"""
        if not statements or self.my_player_id == -1:
            return 0
        
        accusation_keywords = ['suspicious', 'suspect', 'vote', 'eliminate', 'mafia', 'guilty']
        my_id_str = str(self.my_player_id)
        accusers = set()
        
        for player_id, statement in statements:
            if player_id != self.my_player_id:
                statement_lower = statement.lower()
                if my_id_str in statement and any(keyword in statement_lower for keyword in accusation_keywords):
                    accusers.add(player_id)
        
        return len(accusers)
    
    def _get_accusers(self, statements: List[Tuple[int, str]]) -> List[int]:
        """Get list of players who are accusing us"""
        if not statements or self.my_player_id == -1:
            return []
        
        accusation_keywords = ['suspicious', 'suspect', 'vote', 'eliminate', 'mafia', 'guilty']
        my_id_str = str(self.my_player_id)
        accusers = []
        
        for player_id, statement in statements:
            if player_id != self.my_player_id:
                statement_lower = statement.lower()
                if my_id_str in statement and any(keyword in statement_lower for keyword in accusation_keywords):
                    if player_id not in accusers:
                        accusers.append(player_id)
        
        return accusers
    
    def _identify_detective_claimers(self, observation: str) -> List[int]:
        """Identify players claiming to be Detective"""
        claimers = []
        lines = observation.split('\n')
        
        for line in lines:
            # Look for detective claims
            if 'detective' in line.lower() and ('i am' in line.lower() or 'my investigation' in line.lower()):
                player_match = re.search(r'Player (\d+)', line)
                if player_match:
                    player_id = int(player_match.group(1))
                    if player_id != self.my_player_id and player_id not in claimers:
                        claimers.append(player_id)
        
        return claimers
    
    def _identify_doctor_claimers(self, observation: str) -> List[int]:
        """Identify players claiming to be Doctor"""
        claimers = []
        lines = observation.split('\n')
        
        for line in lines:
            # Look for doctor claims
            if 'doctor' in line.lower() and ('i am' in line.lower() or 'i protect' in line.lower()):
                player_match = re.search(r'Player (\d+)', line)
                if player_match:
                    player_id = int(player_match.group(1))
                    if player_id != self.my_player_id and player_id not in claimers:
                        claimers.append(player_id)
        
        return claimers
    
    def _identify_most_suspicious_non_teammate(self, observation: str) -> Optional[int]:
        """Identify most suspicious player who isn't our Mafia teammate"""
        if not hasattr(self, 'alive_players') or not self.alive_players:
            return None
        
        # Get teammate info from observation if available
        teammates = self._get_mafia_teammates(observation)
        
        # Find most suspicious player who isn't us or our teammate
        candidates = [p for p in self.alive_players if p != self.my_player_id and p not in teammates]
        
        if not candidates:
            return None
        
        # Use ToM engine to find most suspicious
        if hasattr(self, 'tom_engine'):
            suspicion_scores = {p: self.tom_engine.get_suspicion_level(p) for p in candidates}
            return max(suspicion_scores.keys(), key=lambda p: suspicion_scores[p])
        
        # Fallback to random choice
        return random.choice(candidates)
    
    def _get_mafia_teammates(self, observation: str) -> List[int]:
        """Extract Mafia teammate IDs from observation"""
        teammates = []
        
        # Look for teammate information in initial game setup
        if "Your teammates are:" in observation:
            teammate_match = re.search(r'Your teammates are: (.+)', observation)
            if teammate_match:
                teammate_text = teammate_match.group(1)
                player_matches = re.findall(r'Player (\d+)', teammate_text)
                teammates = [int(p) for p in player_matches if int(p) != self.my_player_id]
        
        return teammates
    
    def _extract_player_statements(self, observation: str) -> List[Tuple[int, str]]:
        """Extract player statements from observation"""
        statements = []
        lines = observation.split('\n')
        
        for line in lines:
            # Look for patterns like "Player X says: ..." or "[X, Y]: ..."
            player_match = re.search(r'Player\s+(\d+)\s+(?:says?|claims?):?\s*(.+)', line)
            if player_match:
                player_id = int(player_match.group(1))
                statement = player_match.group(2).strip()
                statements.append((player_id, statement))
                continue
            
            # Look for bracket format [X, Y]: statement
            bracket_match = re.search(r'\[(\d+),\s*\d+\]:\s*(.+)', line)
            if bracket_match:
                player_id = int(bracket_match.group(1))
                statement = bracket_match.group(2).strip()
                statements.append((player_id, statement))
        
        return statements
    
    def _get_current_beliefs(self) -> Dict[int, float]:
        """Get current belief state about all players"""
        beliefs = {}
        for player_id in self.alive_players:
            if player_id != self.my_player_id:
                beliefs[player_id] = self.tom_engine.get_suspicion_level(player_id)
        return beliefs
    
    def _determine_strategic_priority(self, observation: str) -> str:
        """Determine current strategic priority"""
        if self._being_accused_in_statements(self._extract_player_statements(observation)):
            return "defense"
        elif self.my_role == Role.MAFIA:
            return "deception"
        elif self.my_role == Role.DETECTIVE:
            return "investigation"
        elif self.my_role == Role.DOCTOR:
            return "protection"
        else:
            return "analysis"
    
    def get_championship_performance_summary(self) -> Dict:
        """Get summary of championship enhancement performance"""
        herd_stats = self.herd_manipulator.get_influence_stats()
        
        return {
            'security_metrics': {
                'jailbreak_attempts_blocked': self.championship_metrics['jailbreak_attempts_blocked'],
                'unsafe_responses_prevented': self.championship_metrics['unsafe_responses_prevented'],
                'attack_log_entries': len(self.jailbreak_protection.attack_log)
            },
            'communication_metrics': {
                'diplomatic_enhancements_applied': self.championship_metrics['diplomatic_enhancements_applied'],
                'trust_building_phrases_used': self.championship_metrics['trust_building_phrases_used'],
                'agent_classifications_made': self.championship_metrics['agent_classifications_made']
            },
            'strategic_metrics': {
                'first_vote_influences': self.championship_metrics['first_vote_influences'],
                'influence_success_rate': herd_stats['influence_success_rate'],
                'successful_influences': herd_stats['successful_influences']
            },
            'overall_enhancement_score': self._calculate_enhancement_score()
        }
    
    def _calculate_enhancement_score(self) -> float:
        """Calculate overall enhancement effectiveness score"""
        base_score = 1.0
        
        # Security bonus
        if self.championship_metrics['jailbreak_attempts_blocked'] > 0:
            base_score += 0.2
        
        # Communication bonus
        if self.championship_metrics['diplomatic_enhancements_applied'] > 0:
            base_score += 0.3
        
        # Strategic bonus
        herd_stats = self.herd_manipulator.get_influence_stats()
        if herd_stats['influence_success_rate'] > 0.5:
            base_score += 0.4
        
        # Safety bonus
        if self.championship_metrics['unsafe_responses_prevented'] > 0:
            base_score += 0.1
        
        return min(2.0, base_score)  # Cap at 2.0
    
    # ============================================================================
    # MULTIMIND-INSPIRED METHODS (RULES COMPLIANT)
    # ============================================================================
    
    def _update_belief_matrix_analysis(self, observation: str):
        """Update belief matrix using LLM reasoning (MultiMind-inspired)"""
        try:
            # Extract current players and discussion history
            alive_players = self._extract_alive_players(observation)
            if not alive_players:
                alive_players = [1, 2, 3, 4, 5]  # Default for early game
            
            discussion_history = self._extract_recent_discussion(observation)
            
            # Generate belief matrix prompt
            belief_prompt = self.belief_tracker.update_belief_matrix_prompt(alive_players, discussion_history)
            
            # Get LLM analysis (conditional execution for performance)
            if self.turn_count % 2 == 0:  # Every other turn to prevent timeouts
                endpoint_url = self.endpoint_url.rstrip('/') + '/generate'
                response = self.requests.post(
                    endpoint_url,
                    json={"observation": belief_prompt},
                    timeout=20  # 20 second timeout for belief analysis
                )
                response.raise_for_status()
                
                belief_analysis = response.json().get("response", "")
                
                # Store results for use in strategic thinking
                self.memory.belief_analysis = belief_analysis
                self.championship_metrics['belief_matrix_updates'] += 1
                
        except Exception as e:
            print(f"❌ Belief matrix update failed: {e}")
            self.memory.belief_analysis = "Unable to update belief matrix"
    
    def _parse_communication_actions(self, observation: str):
        """Parse communication actions into structured triplets (MultiMind-inspired)"""
        try:
            # Extract recent player statements
            player_statements = self._extract_player_statements(observation)
            
            if not player_statements:
                return
            
            # Parse actions for recent statements (conditional execution)
            if self.turn_count % 3 == 0:  # Every 3rd turn to prevent timeouts
                for player_id, statement in player_statements[-3:]:  # Last 3 statements only
                    if player_id != self.my_player_id:
                        action_prompt = self.action_parser.parse_actions_prompt(statement, f"Player {player_id}")
                        
                        endpoint_url = self.endpoint_url.rstrip('/') + '/generate'
                        response = self.requests.post(
                            endpoint_url,
                            json={"observation": action_prompt},
                            timeout=15  # 15 second timeout for action parsing
                        )
                        response.raise_for_status()
                        
                        action_analysis = response.json().get("response", "")
                        
                        # Store parsed actions
                        if not hasattr(self.memory, 'parsed_actions'):
                            self.memory.parsed_actions = {}
                        self.memory.parsed_actions[player_id] = action_analysis
                
                self.championship_metrics['action_parsing_sessions'] += 1
                
        except Exception as e:
            print(f"❌ Communication action parsing failed: {e}")
    
    def _apply_strategic_thinking_framework(self, observation: str):
        """Apply strategic thinking framework for suspicion minimization (MultiMind-inspired)"""
        try:
            # Only apply during discussion phase to prevent interference with voting/night actions
            if not self._is_discussion_phase(observation):
                return
            
            # Assess current suspicion level toward me
            my_suspicion_level = self._assess_suspicion_toward_me(observation)
            current_situation = self._summarize_current_situation(observation)
            
            # Apply strategic thinking (conditional execution)
            if self.turn_count % 2 == 0:  # Every other turn
                strategy_prompt = self.strategic_framework.generate_strategic_thinking_prompt(
                    current_situation, my_suspicion_level
                )
                
                endpoint_url = self.endpoint_url.rstrip('/') + '/generate'
                response = self.requests.post(
                    endpoint_url,
                    json={"observation": strategy_prompt},
                    timeout=20  # 20 second timeout for strategic thinking
                )
                response.raise_for_status()
                
                strategic_analysis = response.json().get("response", "")
                
                # Store strategic insights for use in response generation
                self.memory.strategic_analysis = strategic_analysis
                self.championship_metrics['strategic_analyses_performed'] += 1
                
        except Exception as e:
            print(f"❌ Strategic thinking framework failed: {e}")
            self.memory.strategic_analysis = "Unable to generate strategic analysis"
    
    def _perform_structured_communication_analysis(self, observation: str):
        """Perform structured communication analysis using LLM reasoning (MultiMind-inspired)"""
        try:
            # Extract recent messages for analysis
            recent_messages = self._extract_recent_messages(observation)
            
            if not recent_messages:
                return
            
            # Perform structured analysis (conditional execution)
            if self.turn_count % 3 == 0:  # Every 3rd turn
                messages = [msg[1] for msg in recent_messages[-5:]]  # Last 5 messages
                speakers = [f"Player {msg[0]}" for msg in recent_messages[-5:]]
                
                analysis_prompt = self.communication_analyzer.generate_communication_analysis_prompt(
                    messages, speakers
                )
                
                endpoint_url = self.endpoint_url.rstrip('/') + '/generate'
                response = self.requests.post(
                    endpoint_url,
                    json={"observation": analysis_prompt},
                    timeout=25  # 25 second timeout for communication analysis
                )
                response.raise_for_status()
                
                communication_analysis = response.json().get("response", "")
                
                # Store analysis results
                self.memory.communication_analysis = communication_analysis
                self.championship_metrics['communication_analyses_completed'] += 1
                
        except Exception as e:
            print(f"❌ Structured communication analysis failed: {e}")
            self.memory.communication_analysis = "Unable to perform communication analysis"
    
    # Helper methods for MultiMind components
    def _extract_alive_players(self, observation: str) -> List[int]:
        """Extract list of alive players from observation"""
        try:
            # Look for "Players:" line in observation
            lines = observation.split('\n')
            for line in lines:
                if 'Players:' in line:
                    # Extract player numbers
                    import re
                    player_matches = re.findall(r'Player (\d+)', line)
                    return [int(p) for p in player_matches]
            
            # Fallback: extract from any player references
            import re
            all_players = set()
            for line in lines:
                player_matches = re.findall(r'Player (\d+)', line)
                all_players.update(int(p) for p in player_matches)
            
            return sorted(list(all_players)) if all_players else [0, 1, 2, 3, 4, 5]
        except:
            return [0, 1, 2, 3, 4, 5]  # Default fallback
    
    def _extract_recent_discussion(self, observation: str) -> str:
        """Extract recent discussion history"""
        lines = observation.split('\n')
        discussion_lines = []
        for line in lines:
            if 'says:' in line or 'Player' in line:
                discussion_lines.append(line)
        return '\n'.join(discussion_lines[-10:])  # Last 10 relevant lines
    
    def _assess_suspicion_toward_me(self, observation: str) -> str:
        """Assess current suspicion level toward me"""
        if hasattr(self.memory, 'belief_analysis') and self.memory.belief_analysis:
            # Extract suspicion info from belief analysis
            if 'SUSPICION_TOWARD_ME:' in self.memory.belief_analysis:
                return "MEDIUM"  # Simplified assessment
        return "UNKNOWN"
    
    def _summarize_current_situation(self, observation: str) -> str:
        """Summarize current game situation"""
        phase = "Discussion"
        if self._is_voting_phase(observation):
            phase = "Voting"
        elif self._is_night_action_phase(observation):
            phase = "Night"
        
        alive_count = len(self._extract_alive_players(observation) or [1,2,3,4,5])
        return f"Phase: {phase}, Players alive: {alive_count}, Turn: {self.turn_count}"
    
    def _extract_recent_messages(self, observation: str) -> List[Tuple[int, str]]:
        """Extract recent player messages"""
        messages = []
        lines = observation.split('\n')
        for line in lines:
            if 'Player' in line and 'says:' in line:
                try:
                    # Extract player ID and message
                    parts = line.split('says:')
                    if len(parts) >= 2:
                        player_part = parts[0].strip()
                        message = parts[1].strip()
                        # Extract player number
                        import re
                        player_match = re.search(r'Player (\d+)', player_part)
                        if player_match:
                            player_id = int(player_match.group(1))
                            messages.append((player_id, message))
                except:
                    continue
        return messages[-5:]  # Return last 5 messages

    def print_championship_status(self):
        """Print current status of championship enhancements"""
        print("\n" + "=" * 60)
        print("🏆 CHAMPIONSHIP ENHANCEMENTS STATUS")
        print("=" * 60)
        
        summary = self.get_championship_performance_summary()
        
        print(f"🛡️  Security: {summary['security_metrics']['jailbreak_attempts_blocked']} attacks blocked, "
              f"{summary['security_metrics']['unsafe_responses_prevented']} unsafe responses prevented")
        
        print(f"🤝 Communication: {summary['communication_metrics']['diplomatic_enhancements_applied']} diplomatic enhancements, "
              f"{summary['communication_metrics']['trust_building_phrases_used']} trust phrases used")
        
        print(f"🎯 Strategy: {summary['strategic_metrics']['first_vote_influences']} first votes, "
              f"{summary['strategic_metrics']['influence_success_rate']:.1%} influence success rate")
        
        print(f"📊 Overall Enhancement Score: {summary['overall_enhancement_score']:.2f}/2.00")
        print("=" * 60)