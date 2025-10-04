"""
Strategic Bidding System
Implements research-based dynamic turn-taking with strategic urgency calculation
Based on Werewolf Arena paper findings: 40% of mentioned players bid maximum urgency
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

class UrgencyLevel(Enum):
    OBSERVE = 0      # "I would like to observe and listen for now"
    GENERAL = 1      # "I have some general thoughts to share with the group"
    SPECIFIC = 2     # "I have something critical and specific to contribute"
    URGENT = 3       # "It is absolutely urgent for me to speak next"
    RESPOND = 4      # "Someone has addressed me directly and I must respond"

@dataclass
class GameContext:
    """Current game context for bidding decisions"""
    current_turn: int
    phase: str
    my_role: str
    my_player_id: int
    alive_players: List[int]
    suspicion_on_me: float
    recent_statements: List[Tuple[int, str]]
    voting_phase_active: bool = False
    investigation_results: Optional[Dict] = None
    night_kill_occurred: bool = False

class StrategicBiddingSystem:
    """Research-based strategic bidding for optimal turn-taking"""
    
    def __init__(self):
        self.bid_history: List[Tuple[int, int, str]] = []  # (turn, bid, reason)
        self.mention_tracking: Dict[int, int] = {}  # player_id -> times_mentioned_recently
        self.strategic_silence_turns: int = 0
        
        # Research-based bidding weights
        self.URGENCY_WEIGHTS = {
            'direct_mention': 4,      # 40% of mentioned players bid max (research finding)
            'critical_info': 3,       # Detective results, Mafia accusations
            'high_suspicion': 2,      # When suspicion on self is building
            'strategic_contribution': 2,  # When have valuable analysis
            'defensive_response': 3,   # When being accused
            'role_reveal_timing': 3,   # Optimal timing for role reveals
            'voting_coordination': 2,  # During voting phases
            'silence_break': 1        # Breaking strategic silence
        }
    
    def calculate_strategic_bid(self, context: GameContext, tom_engine=None) -> Tuple[int, str]:
        """Calculate optimal bid based on strategic necessity and research findings"""
        bid_factors = []
        
        # PRIORITY 1: Direct mentions (research: 40% bid maximum urgency)
        if self._was_mentioned_recently(context):
            bid_factors.append(('direct_mention', self.URGENCY_WEIGHTS['direct_mention']))
        
        # PRIORITY 2: Critical information sharing
        critical_info_urgency = self._assess_critical_information(context)
        if critical_info_urgency > 0:
            bid_factors.append(('critical_info', critical_info_urgency))
        
        # PRIORITY 3: Defensive responses to accusations
        if self._being_accused(context):
            bid_factors.append(('defensive_response', self.URGENCY_WEIGHTS['defensive_response']))
        
        # PRIORITY 4: Role-specific strategic timing
        role_urgency = self._calculate_role_specific_urgency(context, tom_engine)
        if role_urgency > 0:
            bid_factors.append(('role_specific', role_urgency))
        
        # PRIORITY 5: Suspicion management
        if context.suspicion_on_me > 0.6:
            bid_factors.append(('high_suspicion', self.URGENCY_WEIGHTS['high_suspicion']))
        
        # PRIORITY 6: Strategic silence vs contribution
        silence_factor = self._evaluate_strategic_silence(context)
        if silence_factor != 1:  # 1 is neutral
            if silence_factor > 1:
                bid_factors.append(('strategic_contribution', silence_factor))
            else:
                bid_factors.append(('strategic_silence', 0))
        
        # PRIORITY 7: Voting phase coordination
        if context.voting_phase_active:
            bid_factors.append(('voting_coordination', self.URGENCY_WEIGHTS['voting_coordination']))
        
        # Calculate final bid
        if not bid_factors:
            # Default bidding with research-based randomization
            return self._default_bid(context)
        
        # Take highest priority factor
        max_factor = max(bid_factors, key=lambda x: x[1])
        bid_value = min(4, max_factor[1])  # Cap at maximum urgency
        
        # Add strategic randomization (research shows some unpredictability helps)
        if random.random() < 0.1:  # 10% chance of strategic variation
            bid_value = max(0, min(4, bid_value + random.choice([-1, 1])))
        
        # Record bid for analysis
        reason = f"{max_factor[0]} (factors: {[f[0] for f in bid_factors]})"
        self.bid_history.append((context.current_turn, bid_value, reason))
        
        return bid_value, reason
    
    def _was_mentioned_recently(self, context: GameContext) -> bool:
        """Check if we were mentioned in recent statements"""
        if not context.recent_statements:
            return False
        
        my_id = context.my_player_id
        mention_patterns = [f"Player {my_id}", f"player {my_id}", f"[{my_id}]"]
        
        # Check last 3 statements for mentions
        recent_statements = context.recent_statements[-3:]
        for speaker_id, statement in recent_statements:
            if speaker_id != my_id:  # Don't count self-mentions
                if any(pattern in statement for pattern in mention_patterns):
                    return True
        
        return False
    
    def _assess_critical_information(self, context: GameContext) -> int:
        """Assess if we have critical information to share"""
        urgency = 0
        
        # Detective with investigation results
        if context.my_role == "Detective" and context.investigation_results:
            if context.investigation_results.get('found_mafia', False):
                urgency = 4  # Maximum urgency for Mafia discovery
            elif context.investigation_results.get('cleared_villager', False):
                urgency = 3  # High urgency for clearing suspicion
        
        # Doctor with protection insights
        elif context.my_role == "Doctor" and context.night_kill_occurred:
            urgency = 2  # Moderate urgency to share kill analysis
        
        # Mafia with strategic misdirection needs
        elif context.my_role == "Mafia":
            # High urgency if town is getting close to truth
            if context.suspicion_on_me > 0.7:
                urgency = 3
            # Moderate urgency for strategic misdirection
            elif self._town_needs_misdirection(context):
                urgency = 2
        
        # Villager with strong analytical insights
        elif context.my_role == "Villager":
            # High urgency if we've identified likely Mafia
            if self._have_strong_mafia_read(context):
                urgency = 3
        
        return urgency
    
    def _being_accused(self, context: GameContext) -> bool:
        """Check if we're being accused in recent statements"""
        if not context.recent_statements:
            return False
        
        my_id = context.my_player_id
        accusation_keywords = ['suspicious', 'mafia', 'doubt', 'lying', 'fake']
        
        for speaker_id, statement in context.recent_statements[-2:]:  # Last 2 statements
            if speaker_id != my_id:
                statement_lower = statement.lower()
                # Check if statement mentions us AND contains accusation keywords
                if (f"player {my_id}" in statement_lower or f"Player {my_id}" in statement):
                    if any(keyword in statement_lower for keyword in accusation_keywords):
                        return True
        
        return False
    
    def _calculate_role_specific_urgency(self, context: GameContext, tom_engine=None) -> int:
        """Calculate role-specific strategic urgency"""
        if context.my_role == "Detective":
            return self._detective_urgency(context, tom_engine)
        elif context.my_role == "Doctor":
            return self._doctor_urgency(context)
        elif context.my_role == "Mafia":
            return self._mafia_urgency(context, tom_engine)
        elif context.my_role == "Villager":
            return self._villager_urgency(context, tom_engine)
        
        return 0
    
    def _detective_urgency(self, context: GameContext, tom_engine=None) -> int:
        """Detective-specific urgency calculation"""
        # High urgency to share investigation results
        if context.investigation_results:
            return 3
        
        # Moderate urgency if multiple Detective claims (need to establish credibility)
        if self._multiple_detective_claims(context):
            return 2
        
        # High urgency if being accused (need to defend role)
        if self._being_accused(context):
            return 3
        
        # Low urgency during early game (strategic concealment)
        if context.current_turn <= 3:
            return 0
        
        return 1
    
    def _doctor_urgency(self, context: GameContext) -> int:
        """Doctor-specific urgency calculation"""
        # Moderate urgency after night kill (share protection insights)
        if context.night_kill_occurred:
            return 2
        
        # High urgency if Detective needs protection coordination
        if self._detective_needs_protection(context):
            return 3
        
        # Low urgency generally (Doctors should stay hidden)
        return 0
    
    def _mafia_urgency(self, context: GameContext, tom_engine=None) -> int:
        """Mafia-specific urgency calculation"""
        # Maximum urgency if about to be voted out
        if context.suspicion_on_me > 0.8:
            return 4
        
        # High urgency if need to deflect from partner
        if tom_engine and self._partner_in_danger(context, tom_engine):
            return 3
        
        # Moderate urgency for strategic misdirection
        if self._optimal_misdirection_timing(context):
            return 2
        
        # Strategic silence when town is confused
        if self._town_is_confused(context):
            return 0
        
        return 1
    
    def _villager_urgency(self, context: GameContext, tom_engine=None) -> int:
        """Villager-specific urgency calculation"""
        # High urgency if have strong Mafia read
        if tom_engine and self._have_strong_mafia_read(context, tom_engine):
            return 3
        
        # Moderate urgency to contribute analysis
        if self._can_contribute_analysis(context):
            return 2
        
        # High urgency if being falsely accused
        if self._being_accused(context) and context.my_role == "Villager":
            return 3
        
        return 1
    
    def _evaluate_strategic_silence(self, context: GameContext) -> int:
        """Evaluate whether strategic silence or contribution is better"""
        # Factors favoring silence
        silence_factors = 0
        
        # Mafia benefits from silence when town is confused
        if context.my_role == "Mafia" and self._town_is_confused(context):
            silence_factors += 2
        
        # Early game silence can be strategic
        if context.current_turn <= 2:
            silence_factors += 1
        
        # High suspicion might warrant silence to avoid digging deeper
        if context.suspicion_on_me > 0.8:
            silence_factors += 1
        
        # Factors favoring contribution
        contribution_factors = 0
        
        # Village roles should generally contribute
        if context.my_role in ["Villager", "Detective", "Doctor"]:
            contribution_factors += 2
        
        # Late game requires more active participation
        if context.current_turn > 5:
            contribution_factors += 1
        
        # Low suspicion allows for safe contribution
        if context.suspicion_on_me < 0.3:
            contribution_factors += 1
        
        # Return net factor (>1 favors contribution, <1 favors silence)
        return max(0, contribution_factors - silence_factors + 1)
    
    def _default_bid(self, context: GameContext) -> Tuple[int, str]:
        """Default bidding strategy with research-based randomization"""
        # Research finding: vary bidding to avoid predictable patterns
        base_probabilities = {
            0: 0.3,  # Observe
            1: 0.4,  # General thoughts
            2: 0.2,  # Specific contribution
            3: 0.08, # Urgent
            4: 0.02  # Direct response
        }
        
        # Adjust probabilities based on role
        if context.my_role == "Mafia":
            # Mafia should be more conservative
            base_probabilities[0] += 0.1
            base_probabilities[1] -= 0.05
            base_probabilities[2] -= 0.05
        elif context.my_role in ["Detective", "Doctor"]:
            # Power roles should be more strategic
            base_probabilities[1] += 0.1
            base_probabilities[2] += 0.1
            base_probabilities[0] -= 0.2
        
        # Random selection based on adjusted probabilities
        rand = random.random()
        cumulative = 0
        for bid, prob in base_probabilities.items():
            cumulative += prob
            if rand <= cumulative:
                return bid, f"default_strategy (role: {context.my_role})"
        
        return 1, "fallback_default"
    
    # Helper methods for strategic assessment
    def _town_needs_misdirection(self, context: GameContext) -> bool:
        """Check if town discussion needs Mafia misdirection"""
        # Simplified heuristic - in real implementation, analyze discussion content
        return context.current_turn > 3 and len(context.recent_statements) > 2
    
    def _have_strong_mafia_read(self, context: GameContext, tom_engine=None) -> bool:
        """Check if we have strong suspicion on likely Mafia"""
        if not tom_engine:
            return False
        
        # Check if any player has very high suspicion (excluding self)
        most_suspicious = tom_engine.get_most_suspicious_players(1, exclude_self_id=context.my_player_id)
        return len(most_suspicious) > 0 and most_suspicious[0][1] > 0.7
    
    def _multiple_detective_claims(self, context: GameContext) -> bool:
        """Check if multiple players have claimed Detective"""
        # Simplified - would need to track role claims in full implementation
        return False
    
    def _detective_needs_protection(self, context: GameContext) -> bool:
        """Check if Detective needs protection coordination"""
        # Simplified heuristic
        return "detective" in str(context.recent_statements).lower()
    
    def _partner_in_danger(self, context: GameContext, tom_engine) -> bool:
        """Check if Mafia partner is in danger"""
        # Would need partner identification in full implementation
        return False
    
    def _optimal_misdirection_timing(self, context: GameContext) -> bool:
        """Check if it's optimal timing for Mafia misdirection"""
        return context.current_turn > 2 and context.suspicion_on_me < 0.5
    
    def _town_is_confused(self, context: GameContext) -> bool:
        """Check if town discussion is confused/unfocused"""
        if not context.recent_statements:
            return True
        
        # Heuristic: if recent statements have many questions but few accusations
        question_count = sum(1 for _, stmt in context.recent_statements if '?' in stmt)
        accusation_keywords = ['suspicious', 'mafia', 'vote']
        accusation_count = sum(1 for _, stmt in context.recent_statements 
                             if any(keyword in stmt.lower() for keyword in accusation_keywords))
        
        return question_count > accusation_count * 2
    
    def _can_contribute_analysis(self, context: GameContext) -> bool:
        """Check if we can contribute meaningful analysis"""
        return len(context.recent_statements) >= 2  # Need some discussion to analyze
    
    def get_bid_explanation(self, bid_value: int) -> str:
        """Get human-readable explanation for bid value"""
        explanations = {
            0: "I would like to observe and listen for now",
            1: "I have some general thoughts to share with the group",
            2: "I have something critical and specific to contribute",
            3: "It is absolutely urgent for me to speak next",
            4: "Someone has addressed me directly and I must respond"
        }
        return explanations.get(bid_value, "Unknown urgency level")
    
    def get_bidding_statistics(self) -> Dict:
        """Get statistics about bidding patterns for analysis"""
        if not self.bid_history:
            return {"message": "No bidding history available"}
        
        bid_counts = {i: 0 for i in range(5)}
        reasons = {}
        
        for turn, bid, reason in self.bid_history:
            bid_counts[bid] += 1
            reason_key = reason.split('(')[0].strip()
            reasons[reason_key] = reasons.get(reason_key, 0) + 1
        
        total_bids = len(self.bid_history)
        
        return {
            "total_bids": total_bids,
            "bid_distribution": {f"level_{k}": v/total_bids for k, v in bid_counts.items()},
            "common_reasons": sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5],
            "average_bid": sum(bid for _, bid, _ in self.bid_history) / total_bids,
            "recent_pattern": [bid for _, bid, _ in self.bid_history[-5:]]  # Last 5 bids
        }