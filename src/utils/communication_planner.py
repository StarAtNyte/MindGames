"""
MCTS-Based Communication Planning
Implements Monte Carlo Tree Search for optimal response selection
Based on MultiMind research: MCTS with 500 iterations significantly outperforms greedy selection
"""

import random
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import copy

@dataclass
class MCTSNode:
    """Node in the MCTS tree"""
    state: Dict[str, Any]
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: List[str] = None
    action_taken: Optional[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            self.untried_actions = []
    
    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node"""
        return len(self.children) == 0 and len(self.untried_actions) == 0

class MCTSCommunicationPlanner:
    """MCTS-based optimal response selection for social deduction"""
    
    def __init__(self, iterations: int = 500, exploration_constant: float = 1.414):
        self.iterations = iterations  # Research shows 500 iterations optimal
        self.exploration_constant = exploration_constant
        self.simulation_cache: Dict[str, float] = {}
        
        # Reward weights based on research findings
        self.REWARD_WEIGHTS = {
            'suspicion_reduction': 2.0,      # Reducing suspicion on self
            'mafia_suspicion_increase': 1.5, # Increasing suspicion on actual Mafia
            'information_value': 1.0,        # Providing valuable information
            'alliance_building': 0.8,        # Building trust with Village
            'deflection_success': 1.2,       # Successfully deflecting (for Mafia)
            'role_concealment': 1.0,         # Keeping role hidden when beneficial
            'voting_influence': 1.3,         # Influencing voting decisions
            'credibility_maintenance': 1.1   # Maintaining believability
        }
    
    def select_optimal_response(self, possible_responses: List[str], game_state: Dict, 
                              tom_engine=None, my_role: str = "Villager") -> Tuple[str, float, Dict]:
        """
        Use MCTS to find response that optimizes strategic objectives
        Returns: (best_response, expected_reward, analysis_details)
        """
        if not possible_responses:
            return "I need more information to decide.", 0.0, {"reason": "no_responses"}
        
        if len(possible_responses) == 1:
            # Single option - still evaluate for analysis
            reward = self._evaluate_response_reward(possible_responses[0], game_state, tom_engine, my_role)
            return possible_responses[0], reward, {"reason": "single_option", "reward": reward}
        
        # Initialize root node
        root_state = self._create_game_state_copy(game_state)
        root = MCTSNode(
            state=root_state,
            untried_actions=possible_responses.copy()
        )
        
        # Run MCTS iterations
        for iteration in range(self.iterations):
            # Selection: traverse tree using UCB1
            node = self._select_node(root)
            
            # Expansion: add new child if not fully expanded
            if not node.is_fully_expanded() and not node.is_terminal():
                node = self._expand_node(node, game_state, tom_engine, my_role)
            
            # Simulation: evaluate the response
            reward = self._simulate_response(node, game_state, tom_engine, my_role)
            
            # Backpropagation: update node values
            self._backpropagate(node, reward)
        
        # Select best response based on visit counts (most robust)
        best_child = max(root.children, key=lambda child: child.visits)
        best_response = best_child.action_taken
        expected_reward = best_child.total_reward / max(1, best_child.visits)
        
        # Generate analysis details
        analysis = self._generate_analysis(root, best_child, game_state)
        
        return best_response, expected_reward, analysis
    
    def _select_node(self, root: MCTSNode) -> MCTSNode:
        """Select node using UCB1 policy"""
        current = root
        
        while not current.is_terminal() and current.is_fully_expanded():
            if not current.children:
                break
            current = max(current.children, key=lambda child: child.ucb1_score(self.exploration_constant))
        
        return current
    
    def _expand_node(self, node: MCTSNode, game_state: Dict, tom_engine, my_role: str) -> MCTSNode:
        """Expand node by adding a new child"""
        if not node.untried_actions:
            return node
        
        # Select random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Create new child node
        child_state = self._simulate_state_after_response(node.state, action, game_state, tom_engine, my_role)
        child = MCTSNode(
            state=child_state,
            parent=node,
            action_taken=action
        )
        
        node.children.append(child)
        return child
    
    def _simulate_response(self, node: MCTSNode, game_state: Dict, tom_engine, my_role: str) -> float:
        """Simulate the outcome of a response and return reward"""
        if not node.action_taken:
            return 0.0
        
        # Use caching for repeated simulations
        cache_key = f"{node.action_taken}_{hash(str(game_state))}"
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]
        
        # Calculate comprehensive reward
        reward = self._evaluate_response_reward(node.action_taken, game_state, tom_engine, my_role)
        
        # Add stochastic element for exploration
        reward += random.gauss(0, 0.1)  # Small random noise
        
        self.simulation_cache[cache_key] = reward
        return reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def _evaluate_response_reward(self, response: str, game_state: Dict, tom_engine, my_role: str) -> float:
        """Comprehensive reward evaluation based on research findings"""
        total_reward = 0.0
        
        # Get current suspicion level on self
        my_player_id = game_state.get('my_player_id', 0)
        current_suspicion = game_state.get('suspicion_on_me', 0.5)
        
        # Simulate belief changes after this response
        if tom_engine:
            simulated_beliefs = tom_engine.simulate_belief_changes(response, my_player_id)
            
            # Reward for reducing suspicion on self
            if my_player_id in simulated_beliefs:
                suspicion_change = simulated_beliefs[my_player_id] - current_suspicion
                if suspicion_change < 0:  # Suspicion decreased
                    total_reward += abs(suspicion_change) * self.REWARD_WEIGHTS['suspicion_reduction']
                else:  # Suspicion increased
                    total_reward -= suspicion_change * self.REWARD_WEIGHTS['suspicion_reduction']
            
            # Reward for increasing suspicion on actual Mafia (if we know who they are)
            if my_role != "Mafia":
                # For non-Mafia, reward suspicion increases on highly suspicious players (excluding self)
                most_suspicious = tom_engine.get_most_suspicious_players(2, exclude_self_id=my_player_id)
                for player_id, suspicion_level in most_suspicious:
                    if player_id in simulated_beliefs and suspicion_level > 0.6:
                        suspicion_increase = simulated_beliefs[player_id] - suspicion_level
                        if suspicion_increase > 0:
                            total_reward += suspicion_increase * self.REWARD_WEIGHTS['mafia_suspicion_increase']
        
        # Analyze response content for strategic value
        response_analysis = self._analyze_response_content(response, game_state, my_role)
        
        # Information value reward
        if response_analysis['provides_information']:
            total_reward += self.REWARD_WEIGHTS['information_value']
        
        # Credibility maintenance
        if response_analysis['maintains_credibility']:
            total_reward += self.REWARD_WEIGHTS['credibility_maintenance']
        else:
            total_reward -= self.REWARD_WEIGHTS['credibility_maintenance'] * 0.5
        
        # Role-specific rewards
        if my_role == "Mafia":
            # Mafia-specific rewards
            if response_analysis['deflects_suspicion']:
                total_reward += self.REWARD_WEIGHTS['deflection_success']
            
            if response_analysis['maintains_cover']:
                total_reward += self.REWARD_WEIGHTS['role_concealment']
            
            # Penalty for being too defensive (suspicious behavior)
            if response_analysis['overly_defensive']:
                total_reward -= 0.8
        
        elif my_role == "Detective":
            # Detective-specific rewards
            if response_analysis['strategic_role_reveal']:
                total_reward += self.REWARD_WEIGHTS['information_value'] * 1.5
            
            if response_analysis['builds_credibility']:
                total_reward += self.REWARD_WEIGHTS['credibility_maintenance'] * 1.2
        
        elif my_role == "Doctor":
            # Doctor-specific rewards
            if response_analysis['protects_role_secrecy']:
                total_reward += self.REWARD_WEIGHTS['role_concealment']
            
            if response_analysis['coordinates_protection']:
                total_reward += self.REWARD_WEIGHTS['alliance_building']
        
        elif my_role == "Villager":
            # Villager-specific rewards
            if response_analysis['contributes_analysis']:
                total_reward += self.REWARD_WEIGHTS['information_value']
            
            if response_analysis['builds_village_alliance']:
                total_reward += self.REWARD_WEIGHTS['alliance_building']
        
        # Voting influence reward
        if game_state.get('voting_phase_active', False):
            if response_analysis['influences_voting']:
                total_reward += self.REWARD_WEIGHTS['voting_influence']
        
        # Penalty for repetitive or low-quality responses
        if response_analysis['repetitive']:
            total_reward -= 0.5
        
        if response_analysis['too_short']:
            total_reward -= 0.3
        
        if response_analysis['too_long']:
            total_reward -= 0.2
        
        return total_reward
    
    def _analyze_response_content(self, response: str, game_state: Dict, my_role: str) -> Dict[str, bool]:
        """Analyze response content for strategic indicators"""
        response_lower = response.lower()
        word_count = len(response.split())
        
        analysis = {
            'provides_information': False,
            'maintains_credibility': True,
            'deflects_suspicion': False,
            'maintains_cover': True,
            'overly_defensive': False,
            'strategic_role_reveal': False,
            'builds_credibility': False,
            'protects_role_secrecy': True,
            'coordinates_protection': False,
            'contributes_analysis': False,
            'builds_village_alliance': False,
            'influences_voting': False,
            'repetitive': False,
            'too_short': word_count < 5,
            'too_long': word_count > 50
        }
        
        # Information provision indicators
        info_keywords = ['because', 'evidence', 'noticed', 'pattern', 'behavior', 'analysis']
        analysis['provides_information'] = any(keyword in response_lower for keyword in info_keywords)
        
        # Credibility indicators
        credible_phrases = ['i think', 'based on', 'my analysis', 'i noticed']
        incredible_phrases = ['trust me', 'believe me', 'i swear', 'obviously']
        
        if any(phrase in response_lower for phrase in incredible_phrases):
            analysis['maintains_credibility'] = False
        elif any(phrase in response_lower for phrase in credible_phrases):
            analysis['builds_credibility'] = True
        
        # Defensive language detection
        defensive_phrases = ['i\'m not', 'that\'s not true', 'you\'re wrong', 'why would i']
        defensive_count = sum(1 for phrase in defensive_phrases if phrase in response_lower)
        analysis['overly_defensive'] = defensive_count >= 2
        
        # Deflection indicators
        deflection_phrases = ['what about', 'but what', 'instead', 'rather than']
        analysis['deflects_suspicion'] = any(phrase in response_lower for phrase in deflection_phrases)
        
        # Role-specific analysis
        if 'detective' in response_lower and 'investigate' in response_lower:
            analysis['strategic_role_reveal'] = my_role == "Detective"
            analysis['protects_role_secrecy'] = my_role != "Detective"
        
        if 'protect' in response_lower and my_role == "Doctor":
            analysis['coordinates_protection'] = True
        
        # Village alliance building
        alliance_phrases = ['we need to', 'let\'s work together', 'village', 'town']
        analysis['builds_village_alliance'] = any(phrase in response_lower for phrase in alliance_phrases)
        
        # Analytical contribution
        analysis_phrases = ['pattern', 'behavior', 'suspicious', 'evidence', 'voting']
        analysis['contributes_analysis'] = any(phrase in response_lower for phrase in analysis_phrases)
        
        # Voting influence
        voting_phrases = ['vote', 'eliminate', 'choose', 'target']
        analysis['influences_voting'] = any(phrase in response_lower for phrase in voting_phrases)
        
        # Check for repetition (simplified)
        recent_responses = game_state.get('recent_responses', [])
        if recent_responses:
            # Simple similarity check
            for recent in recent_responses[-3:]:  # Check last 3 responses
                if self._calculate_similarity(response, recent) > 0.7:
                    analysis['repetitive'] = True
                    break
        
        return analysis
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_game_state_copy(self, game_state: Dict) -> Dict:
        """Create a copy of game state for simulation"""
        return copy.deepcopy(game_state)
    
    def _simulate_state_after_response(self, current_state: Dict, response: str, 
                                     original_game_state: Dict, tom_engine, my_role: str) -> Dict:
        """Simulate game state after making a response"""
        new_state = copy.deepcopy(current_state)
        
        # Update state based on response impact
        if tom_engine:
            my_player_id = original_game_state.get('my_player_id', 0)
            belief_changes = tom_engine.simulate_belief_changes(response, my_player_id)
            
            # Update suspicion levels in state
            for player_id, new_suspicion in belief_changes.items():
                if player_id == my_player_id:
                    new_state['suspicion_on_me'] = new_suspicion
        
        return new_state
    
    def _generate_analysis(self, root: MCTSNode, best_child: MCTSNode, game_state: Dict) -> Dict:
        """Generate detailed analysis of MCTS decision"""
        analysis = {
            'total_iterations': self.iterations,
            'best_response_visits': best_child.visits,
            'best_response_avg_reward': best_child.total_reward / max(1, best_child.visits),
            'exploration_breadth': len(root.children),
            'confidence': best_child.visits / self.iterations,
            'alternative_options': []
        }
        
        # Analyze alternative options
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True)[:3]:
            if child != best_child:
                analysis['alternative_options'].append({
                    'response': child.action_taken,
                    'visits': child.visits,
                    'avg_reward': child.total_reward / max(1, child.visits),
                    'confidence': child.visits / self.iterations
                })
        
        return analysis
    
    def get_planning_statistics(self) -> Dict:
        """Get statistics about MCTS planning performance"""
        return {
            'cache_size': len(self.simulation_cache),
            'iterations_per_decision': self.iterations,
            'exploration_constant': self.exploration_constant,
            'reward_weights': self.REWARD_WEIGHTS
        }
    
    def clear_cache(self):
        """Clear simulation cache to free memory"""
        self.simulation_cache.clear()
    
    def adjust_iterations(self, new_iterations: int):
        """Adjust number of MCTS iterations (for performance tuning)"""
        self.iterations = max(50, min(1000, new_iterations))  # Reasonable bounds