"""
Advanced Theory of Mind Engine
Implements second-order belief modeling based on MultiMind research
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

@dataclass
class BeliefState:
    """Represents belief about a player's suspicion levels"""
    suspicion_level: float = 0.5  # 0.0 = innocent, 1.0 = definitely mafia
    confidence: float = 0.5       # How confident we are in this belief
    evidence_count: int = 0       # Number of evidence pieces
    last_updated: int = 0         # Turn when last updated
    
    def update(self, delta: float, confidence_delta: float = 0.0, turn: int = 0):
        """Update belief with new evidence"""
        self.suspicion_level = max(0.0, min(1.0, self.suspicion_level + delta))
        self.confidence = max(0.0, min(1.0, self.confidence + confidence_delta))
        self.evidence_count += 1
        self.last_updated = turn

@dataclass
class MetaBelief:
    """What I think player A believes about player B"""
    belief_matrix: Dict[int, Dict[int, BeliefState]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(BeliefState)))
    
    def get_belief(self, observer: int, target: int) -> BeliefState:
        """Get what observer believes about target"""
        return self.belief_matrix[observer][target]
    
    def update_belief(self, observer: int, target: int, delta: float, confidence: float = 0.1, turn: int = 0):
        """Update what observer believes about target"""
        self.belief_matrix[observer][target].update(delta, confidence, turn)

class AdvancedToMEngine:
    """Advanced Theory of Mind with second-order belief modeling"""
    
    def __init__(self):
        # First-order beliefs: what I believe about each player
        self.my_beliefs: Dict[int, BeliefState] = defaultdict(BeliefState)
        
        # Second-order beliefs: what I think each player believes about others
        self.meta_beliefs = MetaBelief()
        
        # Behavioral patterns tracking
        self.behavioral_patterns: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Statement analysis cache
        self.statement_cache: Dict[str, Dict] = {}
        
        # Turn counter for temporal analysis
        self.current_turn = 0
    
    def update_beliefs_from_statements(self, statements: List[Tuple[int, str]], voting_patterns: Dict[int, List[int]] = None):
        """Update belief matrices based on player statements and voting"""
        self.current_turn += 1
        
        for speaker_id, statement in statements:
            # Analyze statement for suspicion indicators
            analysis = self._analyze_statement(statement)
            
            # Update first-order beliefs (what I believe)
            self._update_my_beliefs(speaker_id, statement, analysis)
            
            # Update second-order beliefs (what speaker believes about others)
            self._update_meta_beliefs(speaker_id, statement, analysis)
            
            # Track behavioral patterns
            self._update_behavioral_patterns(speaker_id, analysis)
        
        # Incorporate voting patterns if available
        if voting_patterns:
            self._update_beliefs_from_voting(voting_patterns)
    
    def _analyze_statement(self, statement: str) -> Dict:
        """Comprehensive statement analysis"""
        if statement in self.statement_cache:
            return self.statement_cache[statement]
        
        analysis = {
            'mentioned_players': self._extract_mentioned_players(statement),
            'suspicion_indicators': self._extract_suspicion_indicators(statement),
            'defense_indicators': self._extract_defense_indicators(statement),
            'emotion_indicators': self._analyze_emotion(statement),
            'question_count': statement.count('?'),
            'opinion_markers': self._count_opinion_markers(statement),
            'deflection_score': self._calculate_deflection_score(statement),
            'confidence_level': self._assess_confidence(statement),
            'word_count': len(statement.split()),
            'defensive_language': self._detect_defensive_language(statement)
        }
        
        self.statement_cache[statement] = analysis
        return analysis
    
    def _extract_mentioned_players(self, statement: str) -> List[int]:
        """Extract player IDs mentioned in statement"""
        patterns = [
            r'player\s*(\d+)',
            r'\[(\d+)\]',
            r'#(\d+)',
            r'Player\s*(\d+)'
        ]
        
        mentioned = []
        for pattern in patterns:
            matches = re.findall(pattern, statement, re.IGNORECASE)
            mentioned.extend([int(m) for m in matches])
        
        return list(set(mentioned))
    
    def _extract_suspicion_indicators(self, statement: str) -> Dict[int, float]:
        """Extract suspicion levels for mentioned players"""
        suspicion_keywords = {
            'definitely mafia': 0.9,
            'obviously mafia': 0.8,
            'suspicious': 0.6,
            'doubt': 0.4,
            'questionable': 0.3,
            'concerning': 0.3,
            'weird': 0.2,
            'strange': 0.2
        }
        
        mentioned_players = self._extract_mentioned_players(statement)
        suspicions = {}
        
        statement_lower = statement.lower()
        for player_id in mentioned_players:
            suspicion_level = 0.0
            
            # Check for suspicion keywords near player mentions
            for keyword, weight in suspicion_keywords.items():
                if keyword in statement_lower:
                    # Simple proximity check - if keyword is in same statement as player
                    suspicion_level = max(suspicion_level, weight)
            
            if suspicion_level > 0:
                suspicions[player_id] = suspicion_level
        
        return suspicions
    
    def _extract_defense_indicators(self, statement: str) -> Dict[int, float]:
        """Extract defense levels for mentioned players"""
        defense_keywords = {
            'definitely innocent': 0.9,
            'trust': 0.7,
            'innocent': 0.6,
            'believe': 0.4,
            'honest': 0.4,
            'reliable': 0.3,
            'good point': 0.2
        }
        
        mentioned_players = self._extract_mentioned_players(statement)
        defenses = {}
        
        statement_lower = statement.lower()
        for player_id in mentioned_players:
            defense_level = 0.0
            
            for keyword, weight in defense_keywords.items():
                if keyword in statement_lower:
                    defense_level = max(defense_level, weight)
            
            if defense_level > 0:
                defenses[player_id] = defense_level
        
        return defenses
    
    def _analyze_emotion(self, statement: str) -> Dict[str, float]:
        """Analyze emotional content for deception detection"""
        emotion_indicators = {
            'defensive': ['but', 'actually', 'obviously', 'clearly', 'i assure', 'believe me'],
            'aggressive': ['definitely', 'absolutely', 'must be', 'obviously'],
            'uncertain': ['maybe', 'possibly', 'might', 'could be', 'perhaps'],
            'confident': ['i know', 'i saw', 'i\'m certain', 'for sure', 'without doubt'],
            'deflective': ['what about', 'but what', 'instead', 'rather than']
        }
        
        emotions = {}
        statement_lower = statement.lower()
        
        for emotion, keywords in emotion_indicators.items():
            score = sum(1 for keyword in keywords if keyword in statement_lower)
            emotions[emotion] = score / len(keywords)  # Normalize
        
        return emotions
    
    def _count_opinion_markers(self, statement: str) -> int:
        """Count opinion markers in statement"""
        opinion_markers = ['i think', 'i believe', 'my opinion', 'i suspect', 'i feel', 'in my view']
        statement_lower = statement.lower()
        return sum(1 for marker in opinion_markers if marker in statement_lower)
    
    def _calculate_deflection_score(self, statement: str) -> float:
        """Calculate how much the statement deflects vs contributes"""
        deflection_indicators = ['what about', 'but what', 'instead of', 'rather than', 'shouldn\'t we']
        contribution_indicators = ['because', 'evidence', 'noticed', 'pattern', 'behavior']
        
        statement_lower = statement.lower()
        deflection_count = sum(1 for indicator in deflection_indicators if indicator in statement_lower)
        contribution_count = sum(1 for indicator in contribution_indicators if indicator in statement_lower)
        
        if deflection_count + contribution_count == 0:
            return 0.5  # Neutral
        
        return deflection_count / (deflection_count + contribution_count)
    
    def _assess_confidence(self, statement: str) -> float:
        """Assess confidence level in statement"""
        high_confidence = ['definitely', 'certainly', 'absolutely', 'without doubt', 'i know']
        low_confidence = ['maybe', 'possibly', 'might', 'could be', 'perhaps', 'i think']
        
        statement_lower = statement.lower()
        high_count = sum(1 for word in high_confidence if word in statement_lower)
        low_count = sum(1 for word in low_confidence if word in statement_lower)
        
        if high_count > low_count:
            return 0.8
        elif low_count > high_count:
            return 0.3
        else:
            return 0.5
    
    def _detect_defensive_language(self, statement: str) -> float:
        """Detect defensive language patterns"""
        defensive_patterns = [
            'i\'m not', 'i assure', 'believe me', 'trust me', 'i swear',
            'why would i', 'that\'s not true', 'you\'re wrong'
        ]
        
        statement_lower = statement.lower()
        defensive_count = sum(1 for pattern in defensive_patterns if pattern in statement_lower)
        
        return min(1.0, defensive_count * 0.3)  # Cap at 1.0
    
    def _update_my_beliefs(self, speaker_id: int, statement: str, analysis: Dict):
        """Update my beliefs about the speaker based on their statement"""
        # Analyze speaker's behavior for Mafia indicators
        suspicion_delta = 0.0
        
        # High deflection score increases suspicion
        suspicion_delta += analysis['deflection_score'] * 0.2
        
        # Defensive language increases suspicion
        suspicion_delta += analysis['defensive_language'] * 0.3
        
        # High question-to-opinion ratio increases suspicion (classic Mafia behavior)
        if analysis['opinion_markers'] == 0 and analysis['question_count'] > 0:
            suspicion_delta += 0.4  # Only asks questions, never gives opinions
        elif analysis['question_count'] > analysis['opinion_markers'] and analysis['question_count'] >= 2:
            suspicion_delta += 0.2  # More questions than opinions
        
        # Emotional analysis
        emotions = analysis['emotion_indicators']
        if emotions.get('defensive', 0) > 0.3:
            suspicion_delta += 0.2
        if emotions.get('deflective', 0) > 0.3:
            suspicion_delta += 0.15
        
        # Update belief about speaker
        if suspicion_delta != 0:
            self.my_beliefs[speaker_id].update(suspicion_delta, 0.1, self.current_turn)
    
    def _update_meta_beliefs(self, speaker_id: int, statement: str, analysis: Dict):
        """Update what I think the speaker believes about others"""
        # Extract who the speaker is suspicious of or defending
        suspicions = analysis['suspicion_indicators']
        defenses = analysis['defense_indicators']
        
        # Update meta-beliefs based on speaker's expressed opinions
        for target_id, suspicion_level in suspicions.items():
            # Speaker thinks target is suspicious
            delta = (suspicion_level - 0.5) * 0.5  # Scale to reasonable delta
            self.meta_beliefs.update_belief(speaker_id, target_id, delta, 0.2, self.current_turn)
        
        for target_id, defense_level in defenses.items():
            # Speaker thinks target is innocent
            delta = -(defense_level - 0.5) * 0.5  # Negative delta for defense
            self.meta_beliefs.update_belief(speaker_id, target_id, delta, 0.2, self.current_turn)
    
    def _update_behavioral_patterns(self, speaker_id: int, analysis: Dict):
        """Track behavioral patterns for each player"""
        patterns = self.behavioral_patterns[speaker_id]
        
        # Update running averages
        alpha = 0.3  # Learning rate
        patterns['deflection_tendency'] = (1 - alpha) * patterns['deflection_tendency'] + alpha * analysis['deflection_score']
        patterns['defensive_tendency'] = (1 - alpha) * patterns['defensive_tendency'] + alpha * analysis['defensive_language']
        patterns['question_ratio'] = (1 - alpha) * patterns['question_ratio'] + alpha * (
            analysis['question_count'] / max(1, analysis['question_count'] + analysis['opinion_markers'])
        )
        patterns['confidence_level'] = (1 - alpha) * patterns['confidence_level'] + alpha * analysis['confidence_level']
        patterns['activity_level'] = (1 - alpha) * patterns['activity_level'] + alpha * min(1.0, analysis['word_count'] / 50)
    
    def _update_beliefs_from_voting(self, voting_patterns: Dict[int, List[int]]):
        """Update beliefs based on voting patterns"""
        for voter_id, targets in voting_patterns.items():
            for target_id in targets:
                # Voting for someone indicates suspicion
                self.meta_beliefs.update_belief(voter_id, target_id, 0.3, 0.1, self.current_turn)
    
    def get_suspicion_level(self, player_id: int) -> float:
        """Get my current suspicion level for a player"""
        return self.my_beliefs[player_id].suspicion_level
    
    def get_meta_suspicion(self, observer_id: int, target_id: int) -> float:
        """Get what I think observer believes about target"""
        return self.meta_beliefs.get_belief(observer_id, target_id).suspicion_level
    
    def get_behavioral_pattern(self, player_id: int, pattern_type: str) -> float:
        """Get behavioral pattern score for a player"""
        return self.behavioral_patterns[player_id].get(pattern_type, 0.0)
    
    def calculate_alliance_probability(self, player1_id: int, player2_id: int) -> float:
        """Calculate probability that two players are allied (both Mafia)"""
        # Check if they defend each other
        p1_defends_p2 = self.get_meta_suspicion(player1_id, player2_id) < 0.3
        p2_defends_p1 = self.get_meta_suspicion(player2_id, player1_id) < 0.3
        
        # Check behavioral similarity (Mafia often have similar patterns)
        p1_patterns = self.behavioral_patterns[player1_id]
        p2_patterns = self.behavioral_patterns[player2_id]
        
        pattern_similarity = 0.0
        pattern_count = 0
        
        for pattern_type in ['deflection_tendency', 'defensive_tendency', 'question_ratio']:
            if pattern_type in p1_patterns and pattern_type in p2_patterns:
                similarity = 1.0 - abs(p1_patterns[pattern_type] - p2_patterns[pattern_type])
                pattern_similarity += similarity
                pattern_count += 1
        
        if pattern_count > 0:
            pattern_similarity /= pattern_count
        
        # Combine factors
        alliance_score = 0.0
        if p1_defends_p2 and p2_defends_p1:
            alliance_score += 0.4  # Mutual defense
        elif p1_defends_p2 or p2_defends_p1:
            alliance_score += 0.2  # One-way defense
        
        alliance_score += pattern_similarity * 0.3
        
        return min(1.0, alliance_score)
    
    def get_most_suspicious_players(self, n: int = 3, exclude_self_id: int = None) -> List[Tuple[int, float]]:
        """Get the n most suspicious players according to my beliefs (excluding self)"""
        suspicions = [(player_id, belief.suspicion_level) 
                     for player_id, belief in self.my_beliefs.items()
                     if exclude_self_id is None or player_id != exclude_self_id]
        suspicions.sort(key=lambda x: x[1], reverse=True)
        return suspicions[:n]
    
    def get_trust_network(self) -> Dict[int, List[int]]:
        """Get who each player seems to trust (low suspicion)"""
        trust_network = {}
        
        for observer_id in self.meta_beliefs.belief_matrix:
            trusted_players = []
            for target_id, belief in self.meta_beliefs.belief_matrix[observer_id].items():
                if belief.suspicion_level < 0.3:  # Low suspicion = trust
                    trusted_players.append(target_id)
            trust_network[observer_id] = trusted_players
        
        return trust_network
    
    def simulate_belief_changes(self, hypothetical_statement: str, speaker_id: int) -> Dict[int, float]:
        """Simulate how beliefs would change if speaker made this statement"""
        # Create temporary analysis
        temp_analysis = self._analyze_statement(hypothetical_statement)
        
        # Calculate belief changes without actually updating
        belief_changes = {}
        
        # How would this statement affect my belief about the speaker?
        suspicion_delta = 0.0
        suspicion_delta += temp_analysis['deflection_score'] * 0.2
        suspicion_delta += temp_analysis['defensive_language'] * 0.3
        
        if temp_analysis['opinion_markers'] == 0 and temp_analysis['question_count'] > 0:
            suspicion_delta += 0.4
        
        belief_changes[speaker_id] = self.my_beliefs[speaker_id].suspicion_level + suspicion_delta
        
        # How would this affect beliefs about mentioned players?
        for target_id, suspicion_level in temp_analysis['suspicion_indicators'].items():
            current_meta_belief = self.get_meta_suspicion(speaker_id, target_id)
            delta = (suspicion_level - 0.5) * 0.5
            belief_changes[target_id] = current_meta_belief + delta
        
        return belief_changes
    
    def get_comprehensive_strategic_insights(self, my_player_id: int, alive_players: List[int], my_role: str = None) -> str:
        """Generate comprehensive strategic insights for decision making"""
        insights = []
        
        # 1. SUSPICION ANALYSIS
        most_suspicious = self.get_most_suspicious_players(n=3, exclude_self_id=my_player_id)
        if most_suspicious:
            insights.append(f"🎯 TOP SUSPECTS: {[(pid, f'{susp:.2f}') for pid, susp in most_suspicious]}")
        
        # 2. BEHAVIORAL PATTERNS
        behavioral_insights = []
        for player_id in alive_players:
            if player_id == my_player_id:
                continue
            deflection = self.get_behavioral_pattern(player_id, 'deflection_tendency')
            defensive = self.get_behavioral_pattern(player_id, 'defensive_tendency')
            aggression = self.get_behavioral_pattern(player_id, 'aggression_level')
            
            if deflection > 0.6:
                behavioral_insights.append(f"P{player_id}: High deflection ({deflection:.2f})")
            if defensive > 0.6:
                behavioral_insights.append(f"P{player_id}: Very defensive ({defensive:.2f})")
            if aggression > 0.7:
                behavioral_insights.append(f"P{player_id}: Aggressive ({aggression:.2f})")
        
        if behavioral_insights:
            insights.append(f"🧠 BEHAVIOR: {'; '.join(behavioral_insights)}")
        
        # 3. ALLIANCE DETECTION
        alliances = []
        for i, p1 in enumerate(alive_players):
            for p2 in alive_players[i+1:]:
                if p1 != my_player_id and p2 != my_player_id:
                    alliance_prob = self.calculate_alliance_probability(p1, p2)
                    if alliance_prob > 0.6:
                        alliances.append(f"P{p1}-P{p2} ({alliance_prob:.2f})")
        
        if alliances:
            insights.append(f"🤝 ALLIANCES: {'; '.join(alliances)}")
        
        # 4. TRUST NETWORK
        trust_network = self.get_trust_network()
        trust_insights = []
        for observer, trusted in trust_network.items():
            if observer in alive_players and observer != my_player_id:
                if trusted:
                    trust_insights.append(f"P{observer} trusts: {trusted}")
        
        if trust_insights:
            insights.append(f"💝 TRUST: {'; '.join(trust_insights[:2])}")  # Limit to top 2
        
        # 5. META-BELIEF ANALYSIS (what others think about others)
        meta_insights = []
        for observer in alive_players:
            if observer == my_player_id:
                continue
            observer_suspicions = []
            for target in alive_players:
                if target != observer and target != my_player_id:
                    meta_suspicion = self.get_meta_suspicion(observer, target)
                    if meta_suspicion > 0.6:
                        observer_suspicions.append(f"P{target}({meta_suspicion:.2f})")
            
            if observer_suspicions:
                meta_insights.append(f"P{observer} suspects: {', '.join(observer_suspicions[:2])}")
        
        if meta_insights:
            insights.append(f"👁️ META-BELIEFS: {'; '.join(meta_insights[:2])}")
        
        # 6. ROLE-SPECIFIC INSIGHTS
        if my_role:
            role_insights = self._get_role_specific_tom_insights(my_role, alive_players, my_player_id)
            if role_insights:
                insights.append(f"🎭 ROLE-SPECIFIC: {role_insights}")
        
        return "\n".join(insights) if insights else "No significant patterns detected"
    
    def _get_role_specific_tom_insights(self, my_role: str, alive_players: List[int], my_player_id: int) -> str:
        """Get role-specific Theory of Mind insights"""
        if my_role.lower() == "mafia":
            # Find who's most suspicious of me
            threats = []
            for player in alive_players:
                if player != my_player_id:
                    their_suspicion_of_me = self.get_meta_suspicion(player, my_player_id)
                    if their_suspicion_of_me > 0.5:
                        threats.append(f"P{player}({their_suspicion_of_me:.2f})")
            return f"Threats to me: {', '.join(threats)}" if threats else "No immediate threats"
        
        elif my_role.lower() == "detective":
            # Find who would be good investigation targets
            investigation_targets = []
            for player in alive_players:
                if player != my_player_id:
                    suspicion = self.get_suspicion_level(player)
                    confidence = self.my_beliefs[player].confidence
                    if suspicion > 0.4 and confidence < 0.7:  # Suspicious but uncertain
                        investigation_targets.append(f"P{player}(s:{suspicion:.2f},c:{confidence:.2f})")
            return f"Good investigation targets: {', '.join(investigation_targets[:3])}"
        
        elif my_role.lower() == "doctor":
            # Find who's most likely to be targeted by Mafia
            protection_priorities = []
            for player in alive_players:
                if player != my_player_id:
                    # High value targets: low suspicion (likely villager) + high activity
                    suspicion = self.get_suspicion_level(player)
                    if suspicion < 0.3:  # Likely innocent
                        protection_priorities.append(f"P{player}({suspicion:.2f})")
            return f"Protection priorities: {', '.join(protection_priorities[:3])}"
        
        return ""