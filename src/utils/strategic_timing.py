"""
Strategic Timing System for Mafia Agent
Addresses critical timing issues identified in performance analysis:
- Poor role reveal timing (Detective reveals leading to immediate elimination)
- Analysis paralysis (over-analyzing simple situations)
- Missed strategic opportunities
- Poor adaptation to changing game dynamics
"""

import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class GamePhase(Enum):
    EARLY_GAME = "early"  # Turns 1-3
    MID_GAME = "mid"      # Turns 4-6  
    LATE_GAME = "late"    # Turns 7+
    CRITICAL = "critical"  # Final 3-4 players

class DecisionType(Enum):
    ROLE_REVEAL = "role_reveal"
    ACCUSATION = "accusation"
    INFORMATION_SHARE = "info_share"
    ALLIANCE_FORMATION = "alliance"
    VOTE_DECISION = "vote"
    DEFENSIVE_RESPONSE = "defense"

class TimingContext(Enum):
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    RISKY = "risky"
    DANGEROUS = "dangerous"

@dataclass
class GameStateSnapshot:
    """Snapshot of current game state for timing decisions"""
    turn_number: int
    alive_players: int
    confirmed_mafia: int
    confirmed_village: int
    suspicion_on_me: float
    recent_deaths: bool
    voting_phase_active: bool
    time_pressure: bool
    my_role: str
    my_safety_level: float

@dataclass
class TimingDecision:
    """A timing decision with context and rationale"""
    decision_type: DecisionType
    timing_context: TimingContext
    delay_turns: int
    trigger_conditions: List[str]
    risk_assessment: float
    expected_outcome: str
    backup_plans: List[str]

class StrategicTimingEngine:
    """Advanced timing engine for optimal decision making"""
    
    def __init__(self):
        # ROLE-SPECIFIC TIMING STRATEGIES
        self.role_timing_strategies = {
            'detective': {
                'role_reveal': {
                    'early_game': {'delay_turns': 2, 'risk_threshold': 0.3},
                    'mid_game': {'delay_turns': 1, 'risk_threshold': 0.5},
                    'late_game': {'delay_turns': 0, 'risk_threshold': 0.7},
                    'critical': {'delay_turns': 0, 'risk_threshold': 0.9}
                },
                'information_share': {
                    'early_game': {'delay_turns': 1, 'risk_threshold': 0.2},
                    'mid_game': {'delay_turns': 0, 'risk_threshold': 0.4},
                    'late_game': {'delay_turns': 0, 'risk_threshold': 0.6},
                    'critical': {'delay_turns': 0, 'risk_threshold': 0.8}
                }
            },
            'doctor': {
                'role_reveal': {
                    'early_game': {'delay_turns': 3, 'risk_threshold': 0.2},
                    'mid_game': {'delay_turns': 2, 'risk_threshold': 0.4},
                    'late_game': {'delay_turns': 1, 'risk_threshold': 0.6},
                    'critical': {'delay_turns': 0, 'risk_threshold': 0.8}
                }
            },
            'villager': {
                'accusation': {
                    'early_game': {'delay_turns': 1, 'risk_threshold': 0.4},
                    'mid_game': {'delay_turns': 0, 'risk_threshold': 0.5},
                    'late_game': {'delay_turns': 0, 'risk_threshold': 0.6},
                    'critical': {'delay_turns': 0, 'risk_threshold': 0.7}
                }
            },
            'mafia': {
                'defensive_response': {
                    'early_game': {'delay_turns': 0, 'risk_threshold': 0.3},
                    'mid_game': {'delay_turns': 0, 'risk_threshold': 0.4},
                    'late_game': {'delay_turns': 0, 'risk_threshold': 0.5},
                    'critical': {'delay_turns': 0, 'risk_threshold': 0.6}
                }
            }
        }
        
        # OPTIMAL TIMING CONDITIONS for different decisions
        self.optimal_conditions = {
            DecisionType.ROLE_REVEAL: {
                'detective': [
                    'confirmed_mafia_found',
                    'high_trust_established',
                    'protection_available',
                    'critical_game_state',
                    'village_losing'
                ],
                'doctor': [
                    'multiple_village_confirmed',
                    'mafia_targeting_pattern_clear',
                    'endgame_situation',
                    'vote_coordination_needed'
                ]
            },
            DecisionType.ACCUSATION: [
                'strong_evidence_gathered',
                'alliance_support_available',
                'target_isolated',
                'vote_coordination_ready'
            ],
            DecisionType.INFORMATION_SHARE: [
                'trust_established',
                'information_actionable',
                'timing_not_harmful',
                'group_receptive'
            ],
            DecisionType.ALLIANCE_FORMATION: [
                'mutual_benefit_clear',
                'trust_signals_present',
                'low_suspicion_environment',
                'strategic_value_high'
            ],
            DecisionType.VOTE_DECISION: [
                'information_sufficient',
                'group_consensus_building',
                'strategic_timing_optimal'
            ],
            DecisionType.DEFENSIVE_RESPONSE: [
                'accusation_serious',
                'evidence_available',
                'group_attention_focused',
                'deflection_opportunity_present'
            ]
        }
        
        # DANGER SIGNALS that should delay or modify decisions
        self.danger_signals = {
            'high_suspicion_on_me': 0.8,
            'recent_role_reveal_failed': True,
            'mafia_control_voting': True,
            'information_leak_risk': 0.7,
            'alliance_betrayal_risk': 0.6,
            'endgame_desperation': True
        }
        
        # ADAPTIVE TRIGGERS for changing strategies
        self.adaptive_triggers = {
            'village_losing_badly': {
                'condition': 'mafia_advantage_clear',
                'adjustments': {
                    'role_reveal': {'risk_threshold': -0.2, 'delay_turns': -1},
                    'accusation': {'risk_threshold': -0.1, 'delay_turns': 0},
                    'information_share': {'risk_threshold': -0.1, 'delay_turns': -1}
                }
            },
            'village_winning': {
                'condition': 'village_advantage_clear',
                'adjustments': {
                    'role_reveal': {'risk_threshold': 0.1, 'delay_turns': 1},
                    'information_share': {'risk_threshold': 0.1, 'delay_turns': 0}
                }
            },
            'high_chaos': {
                'condition': 'multiple_accusations_flying',
                'adjustments': {
                    'accusation': {'risk_threshold': 0.2, 'delay_turns': 1},
                    'defensive_response': {'risk_threshold': -0.1, 'delay_turns': 0}
                }
            }
        }
        
        # CONTINGENCY PLANS for when timing goes wrong
        self.contingency_plans = {
            'role_reveal_backfired': [
                'immediate_evidence_presentation',
                'alliance_activation',
                'counter_accusation',
                'information_dump_strategy'
            ],
            'accusation_failed': [
                'graceful_backdown',
                'evidence_reframing',
                'alliance_consultation',
                'suspicion_redirection'
            ],
            'alliance_betrayed': [
                'trust_network_activation',
                'evidence_against_betrayer',
                'damage_control_mode',
                'survival_strategy'
            ],
            'information_leaked': [
                'damage_assessment',
                'counter_information_strategy',
                'alliance_realignment',
                'defensive_positioning'
            ]
        }
        
        self.decision_history: List[TimingDecision] = []
        self.current_adaptations: Dict[str, float] = {}
    
    def evaluate_decision_timing(self, decision_type: DecisionType, game_state: GameStateSnapshot) -> TimingDecision:
        """Evaluate optimal timing for a specific decision"""
        
        # Determine current game phase
        game_phase = self._determine_game_phase(game_state)
        
        # Get base timing strategy for role and decision type
        base_strategy = self._get_base_timing_strategy(decision_type, game_state.my_role, game_phase)
        
        # Analyze current conditions
        condition_analysis = self._analyze_current_conditions(decision_type, game_state)
        
        # Check for danger signals
        danger_assessment = self._assess_danger_signals(game_state)
        
        # Apply adaptive adjustments
        adaptive_adjustments = self._calculate_adaptive_adjustments(game_state)
        
        # Make final timing recommendation
        timing_decision = self._synthesize_timing_decision(
            decision_type, base_strategy, condition_analysis, 
            danger_assessment, adaptive_adjustments, game_state
        )
        
        # Store decision for learning
        self.decision_history.append(timing_decision)
        
        return timing_decision
    
    def should_act_now(self, decision_type: DecisionType, game_state: GameStateSnapshot) -> Tuple[bool, str]:
        """Quick decision on whether to act immediately or wait"""
        
        timing_decision = self.evaluate_decision_timing(decision_type, game_state)
        
        should_act = (
            timing_decision.timing_context in [TimingContext.OPTIMAL, TimingContext.ACCEPTABLE] or
            timing_decision.delay_turns == 0 or
            game_state.time_pressure
        )
        
        reasoning = self._explain_timing_reasoning(timing_decision, game_state)
        
        return should_act, reasoning
    
    def get_pressure_response_strategy(self, accusation_severity: float, game_state: GameStateSnapshot) -> Dict[str, Any]:
        """Generate strategy for responding to pressure/accusations"""
        
        strategy = {
            'response_timing': 'immediate',
            'response_type': 'calm_defense',
            'deflection_targets': [],
            'evidence_to_present': [],
            'alliance_activation': False,
            'contingency_plan': 'damage_control'
        }
        
        # Adjust based on accusation severity
        if accusation_severity > 0.8:
            strategy['response_timing'] = 'immediate'
            strategy['response_type'] = 'strong_defense'
            strategy['alliance_activation'] = True
        elif accusation_severity > 0.6:
            strategy['response_timing'] = 'quick'
            strategy['response_type'] = 'evidence_based_defense'
        else:
            strategy['response_timing'] = 'measured'
            strategy['response_type'] = 'calm_clarification'
        
        # Role-specific adjustments
        if game_state.my_role == 'detective' and game_state.confirmed_mafia > 0:
            strategy['evidence_to_present'].append('investigation_results')
        elif game_state.my_role == 'doctor':
            strategy['response_type'] = 'protective_focus'
        elif game_state.my_role == 'mafia':
            strategy['deflection_targets'] = self._identify_deflection_targets(game_state)
        
        # Context-specific adjustments
        if game_state.voting_phase_active:
            strategy['response_timing'] = 'immediate'
            strategy['contingency_plan'] = 'vote_coordination'
        
        return strategy
    
    def optimize_information_release(self, information_value: float, game_state: GameStateSnapshot) -> Dict[str, Any]:
        """Optimize timing and method of information release"""
        
        release_strategy = {
            'timing': 'hold',
            'method': 'direct_statement',
            'audience': 'general',
            'packaging': 'analytical',
            'support_needed': False
        }
        
        # High-value information needs careful timing
        if information_value > 0.8:
            if game_state.my_safety_level > 0.6:
                release_strategy['timing'] = 'optimal_moment'
                release_strategy['support_needed'] = True
            else:
                release_strategy['timing'] = 'emergency_only'
        elif information_value > 0.6:
            release_strategy['timing'] = 'when_safe'
            release_strategy['method'] = 'gradual_revelation'
        else:
            release_strategy['timing'] = 'when_relevant'
        
        # Game phase adjustments
        game_phase = self._determine_game_phase(game_state)
        if game_phase == GamePhase.CRITICAL:
            release_strategy['timing'] = 'immediate'
            release_strategy['method'] = 'direct_statement'
        elif game_phase == GamePhase.EARLY_GAME:
            release_strategy['timing'] = 'build_trust_first'
            release_strategy['packaging'] = 'collaborative'
        
        return release_strategy
    
    def _determine_game_phase(self, game_state: GameStateSnapshot) -> GamePhase:
        """Determine current game phase based on state"""
        
        if game_state.alive_players <= 4:
            return GamePhase.CRITICAL
        elif game_state.turn_number >= 7:
            return GamePhase.LATE_GAME
        elif game_state.turn_number >= 4:
            return GamePhase.MID_GAME
        else:
            return GamePhase.EARLY_GAME
    
    def _get_base_timing_strategy(self, decision_type: DecisionType, role: str, game_phase: GamePhase) -> Dict[str, Any]:
        """Get base timing strategy for role and decision type"""
        
        role_strategies = self.role_timing_strategies.get(role, {})
        decision_strategies = role_strategies.get(decision_type.value, {})
        phase_strategy = decision_strategies.get(game_phase.value, {'delay_turns': 0, 'risk_threshold': 0.5})
        
        return phase_strategy
    
    def _analyze_current_conditions(self, decision_type: DecisionType, game_state: GameStateSnapshot) -> Dict[str, bool]:
        """Analyze if optimal conditions are met for the decision"""
        
        optimal_conditions = self.optimal_conditions.get(decision_type, [])
        
        # For role-specific decisions, get role-specific conditions
        if decision_type == DecisionType.ROLE_REVEAL:
            optimal_conditions = self.optimal_conditions[decision_type].get(game_state.my_role, [])
        
        condition_status = {}
        
        for condition in optimal_conditions:
            condition_status[condition] = self._evaluate_condition(condition, game_state)
        
        return condition_status
    
    def _evaluate_condition(self, condition: str, game_state: GameStateSnapshot) -> bool:
        """Evaluate whether a specific condition is met"""
        
        condition_evaluators = {
            'confirmed_mafia_found': lambda gs: gs.confirmed_mafia > 0,
            'high_trust_established': lambda gs: gs.suspicion_on_me < 0.3,
            'protection_available': lambda gs: gs.my_role == 'doctor' or gs.confirmed_village > 1,
            'critical_game_state': lambda gs: gs.alive_players <= 4,
            'village_losing': lambda gs: gs.confirmed_mafia >= gs.confirmed_village,
            'strong_evidence_gathered': lambda gs: gs.confirmed_mafia > 0 or gs.turn_number > 3,
            'alliance_support_available': lambda gs: gs.confirmed_village > 0,
            'target_isolated': lambda gs: True,  # Would need specific target info
            'vote_coordination_ready': lambda gs: gs.voting_phase_active,
            'trust_established': lambda gs: gs.suspicion_on_me < 0.4,
            'information_actionable': lambda gs: gs.voting_phase_active or gs.turn_number > 2,
            'timing_not_harmful': lambda gs: gs.suspicion_on_me < 0.6,
            'group_receptive': lambda gs: not gs.recent_deaths,
            'mutual_benefit_clear': lambda gs: gs.turn_number > 2,
            'trust_signals_present': lambda gs: gs.suspicion_on_me < 0.5,
            'low_suspicion_environment': lambda gs: gs.suspicion_on_me < 0.4,
            'strategic_value_high': lambda gs: gs.alive_players > 4,
            'information_sufficient': lambda gs: gs.turn_number > 2,
            'group_consensus_building': lambda gs: gs.voting_phase_active,
            'strategic_timing_optimal': lambda gs: True,  # Context-dependent
            'accusation_serious': lambda gs: gs.suspicion_on_me > 0.6,
            'evidence_available': lambda gs: gs.turn_number > 1,
            'group_attention_focused': lambda gs: gs.suspicion_on_me > 0.5,
            'deflection_opportunity_present': lambda gs: gs.confirmed_mafia == 0
        }
        
        evaluator = condition_evaluators.get(condition, lambda gs: False)
        return evaluator(game_state)
    
    def _assess_danger_signals(self, game_state: GameStateSnapshot) -> Dict[str, Any]:
        """Assess danger signals that might affect timing"""
        
        danger_assessment = {
            'overall_danger_level': 0.0,
            'specific_dangers': [],
            'recommended_adjustments': {}
        }
        
        # Check each danger signal
        if game_state.suspicion_on_me > self.danger_signals['high_suspicion_on_me']:
            danger_assessment['overall_danger_level'] += 0.3
            danger_assessment['specific_dangers'].append('high_suspicion')
        
        if game_state.recent_deaths:
            danger_assessment['overall_danger_level'] += 0.2
            danger_assessment['specific_dangers'].append('recent_deaths')
        
        if game_state.alive_players <= 4:
            danger_assessment['overall_danger_level'] += 0.1
            danger_assessment['specific_dangers'].append('endgame_pressure')
        
        # Generate recommended adjustments based on dangers
        if danger_assessment['overall_danger_level'] > 0.5:
            danger_assessment['recommended_adjustments'] = {
                'increase_caution': True,
                'delay_risky_decisions': True,
                'activate_contingencies': True
            }
        
        return danger_assessment
    
    def _calculate_adaptive_adjustments(self, game_state: GameStateSnapshot) -> Dict[str, float]:
        """Calculate adaptive adjustments based on game state"""
        
        adjustments = {'risk_threshold': 0.0, 'delay_turns': 0}
        
        # Check adaptive triggers
        for trigger_name, trigger_config in self.adaptive_triggers.items():
            if self._evaluate_adaptive_condition(trigger_config['condition'], game_state):
                trigger_adjustments = trigger_config['adjustments']
                for adjustment_type, adjustment_value in trigger_adjustments.items():
                    if adjustment_type in adjustments:
                        adjustments[adjustment_type] += adjustment_value.get('risk_threshold', 0)
        
        return adjustments
    
    def _evaluate_adaptive_condition(self, condition: str, game_state: GameStateSnapshot) -> bool:
        """Evaluate adaptive condition"""
        
        condition_evaluators = {
            'mafia_advantage_clear': lambda gs: gs.confirmed_mafia > gs.confirmed_village,
            'village_advantage_clear': lambda gs: gs.confirmed_village > gs.confirmed_mafia,
            'multiple_accusations_flying': lambda gs: gs.suspicion_on_me > 0.5 and gs.recent_deaths
        }
        
        evaluator = condition_evaluators.get(condition, lambda gs: False)
        return evaluator(game_state)
    
    def _synthesize_timing_decision(self, decision_type: DecisionType, base_strategy: Dict, 
                                   condition_analysis: Dict, danger_assessment: Dict, 
                                   adaptive_adjustments: Dict, game_state: GameStateSnapshot) -> TimingDecision:
        """Synthesize all factors into final timing decision"""
        
        # Calculate final risk threshold
        base_risk_threshold = base_strategy.get('risk_threshold', 0.5)
        adjusted_risk_threshold = base_risk_threshold + adaptive_adjustments.get('risk_threshold', 0.0)
        
        # Calculate delay
        base_delay = base_strategy.get('delay_turns', 0)
        danger_delay = 1 if danger_assessment['overall_danger_level'] > 0.6 else 0
        final_delay = max(0, base_delay + danger_delay + adaptive_adjustments.get('delay_turns', 0))
        
        # Determine timing context
        conditions_met = sum(condition_analysis.values())
        total_conditions = len(condition_analysis)
        
        if conditions_met >= total_conditions * 0.8 and danger_assessment['overall_danger_level'] < 0.3:
            timing_context = TimingContext.OPTIMAL
        elif conditions_met >= total_conditions * 0.6 and danger_assessment['overall_danger_level'] < 0.5:
            timing_context = TimingContext.ACCEPTABLE
        elif danger_assessment['overall_danger_level'] > 0.7:
            timing_context = TimingContext.DANGEROUS
        else:
            timing_context = TimingContext.RISKY
        
        # Generate trigger conditions
        trigger_conditions = [condition for condition, met in condition_analysis.items() if not met]
        
        # Calculate risk assessment
        risk_assessment = min(1.0, danger_assessment['overall_danger_level'] + (1.0 - adjusted_risk_threshold))
        
        # Generate expected outcome
        expected_outcome = self._predict_outcome(decision_type, timing_context, game_state)
        
        # Select appropriate backup plans
        backup_plans = self._select_backup_plans(decision_type, timing_context)
        
        return TimingDecision(
            decision_type=decision_type,
            timing_context=timing_context,
            delay_turns=final_delay,
            trigger_conditions=trigger_conditions,
            risk_assessment=risk_assessment,
            expected_outcome=expected_outcome,
            backup_plans=backup_plans
        )
    
    def _predict_outcome(self, decision_type: DecisionType, timing_context: TimingContext, game_state: GameStateSnapshot) -> str:
        """Predict likely outcome of decision based on timing"""
        
        outcome_predictions = {
            (DecisionType.ROLE_REVEAL, TimingContext.OPTIMAL): "High chance of acceptance and protection",
            (DecisionType.ROLE_REVEAL, TimingContext.ACCEPTABLE): "Moderate chance of success",
            (DecisionType.ROLE_REVEAL, TimingContext.RISKY): "May face skepticism or targeting",
            (DecisionType.ROLE_REVEAL, TimingContext.DANGEROUS): "High risk of immediate elimination",
            
            (DecisionType.ACCUSATION, TimingContext.OPTIMAL): "Strong chance of successful elimination",
            (DecisionType.ACCUSATION, TimingContext.ACCEPTABLE): "Moderate support expected",
            (DecisionType.ACCUSATION, TimingContext.RISKY): "May face counter-accusations",
            (DecisionType.ACCUSATION, TimingContext.DANGEROUS): "Likely to backfire",
        }
        
        return outcome_predictions.get((decision_type, timing_context), "Uncertain outcome")
    
    def _select_backup_plans(self, decision_type: DecisionType, timing_context: TimingContext) -> List[str]:
        """Select appropriate backup plans based on decision and timing"""
        
        if timing_context in [TimingContext.RISKY, TimingContext.DANGEROUS]:
            return self.contingency_plans.get(f"{decision_type.value}_failed", ["damage_control"])
        else:
            return ["monitor_reactions", "be_ready_to_adapt"]
    
    def _explain_timing_reasoning(self, timing_decision: TimingDecision, game_state: GameStateSnapshot) -> str:
        """Generate human-readable explanation of timing reasoning"""
        
        reasoning_parts = []
        
        # Timing context explanation
        context_explanations = {
            TimingContext.OPTIMAL: "Conditions are ideal for this action",
            TimingContext.ACCEPTABLE: "Conditions are reasonable for this action", 
            TimingContext.RISKY: "Conditions are suboptimal but action may be necessary",
            TimingContext.DANGEROUS: "Conditions are poor, high risk of negative outcome"
        }
        
        reasoning_parts.append(context_explanations[timing_decision.timing_context])
        
        # Delay explanation
        if timing_decision.delay_turns > 0:
            reasoning_parts.append(f"Recommend waiting {timing_decision.delay_turns} turn(s)")
        else:
            reasoning_parts.append("Action can be taken immediately")
        
        # Risk explanation
        if timing_decision.risk_assessment > 0.7:
            reasoning_parts.append("High risk situation - proceed with extreme caution")
        elif timing_decision.risk_assessment > 0.5:
            reasoning_parts.append("Moderate risk - ensure backup plans are ready")
        
        return ". ".join(reasoning_parts)
    
    def _identify_deflection_targets(self, game_state: GameStateSnapshot) -> List[int]:
        """Identify potential deflection targets for Mafia role"""
        
        # This would analyze other players to find good deflection targets
        # For now, return empty list as placeholder
        return []