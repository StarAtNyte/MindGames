"""
Modal Labs Data Preprocessing Pipeline for Mafia Game Fine-tuning
Processes game logs for training Qwen 2.5 7B on social deduction gameplay

Key Features:
- Distributed preprocessing on Modal Labs infrastructure
- Rules-compliant data extraction (no heuristics labels)
- Strategic reasoning pattern identification
- Multi-format training data generation (SFT + DPO)
"""

import json
import os
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import modal

# Modal app configuration
app = modal.App("mafia-training-preprocessing")

# Modal image with required dependencies
image = modal.Image.debian_slim().pip_install([
    "torch>=2.0.0",
    "transformers>=4.35.0", 
    "datasets>=2.14.0",
    "jsonlines",
    "pandas",
    "numpy",
    "scikit-learn"
])

# Modal volume for persistent storage
volume = modal.Volume.from_name("mafia-training-data", create_if_missing=True)

@dataclass
class GameLogEntry:
    """Structured game log entry for training"""
    game_id: str
    turn_number: int
    player_role: str
    game_phase: str  # discussion, vote, night, result
    observation: str
    action_taken: str
    reasoning_context: str
    strategic_thinking: str
    outcome: str  # win, loss, ongoing
    suspicion_level: str
    alliance_info: str

@dataclass
class TrainingExample:
    """Training example for fine-tuning"""
    instruction: str
    input_context: str
    target_response: str
    reasoning_chain: str
    quality_score: float
    example_type: str  # sft, dpo_chosen, dpo_rejected

class MafiaGameLogProcessor:
    """Process Mafia game logs for training data generation"""
    
    def __init__(self):
        self.role_patterns = {
            'detective': r'You are Player (\d+), a Detective',
            'doctor': r'You are Player (\d+), a Doctor', 
            'villager': r'You are Player (\d+), a Villager',
            'mafia': r'You are Player (\d+), a Mafia'
        }
        
        self.phase_patterns = {
            'discussion': r'Discussion Phase',
            'vote': r'Vote Phase|Please vote',
            'night': r'Night Phase',
            'result': r'Game Over|wins|eliminated'
        }
        
        # Strategic reasoning indicators for quality assessment
        self.strategic_indicators = [
            'analyze', 'suspect', 'trust', 'evidence', 'pattern',
            'alliance', 'strategy', 'reasoning', 'logic', 'deduce',
            'investigate', 'protect', 'coordinate', 'timing', 'reveal'
        ]
        
        # Quality filters
        self.min_reasoning_length = 50
        self.min_strategic_indicators = 2

    def extract_game_entries(self, game_log: Dict[str, Any]) -> List[GameLogEntry]:
        """Extract structured entries from raw game log"""
        entries = []
        
        try:
            game_id = game_log.get('game_id', 'unknown')
            player_role = self._extract_player_role(game_log)
            outcome = self._determine_outcome(game_log)
            
            # Process each turn in the game
            history = game_log.get('history', [])
            for turn_idx, turn_data in enumerate(history):
                observation = turn_data.get('observation', '')
                action = turn_data.get('response', '')
                
                if not observation or not action:
                    continue
                
                # Extract game phase
                game_phase = self._extract_game_phase(observation)
                
                # Extract strategic reasoning
                reasoning_context = self._extract_reasoning_context(observation, action)
                strategic_thinking = self._extract_strategic_thinking(action)
                
                # Extract social dynamics
                suspicion_level = self._extract_suspicion_level(observation)
                alliance_info = self._extract_alliance_info(observation, action)
                
                entry = GameLogEntry(
                    game_id=game_id,
                    turn_number=turn_idx + 1,
                    player_role=player_role,
                    game_phase=game_phase,
                    observation=observation,
                    action_taken=action,
                    reasoning_context=reasoning_context,
                    strategic_thinking=strategic_thinking,
                    outcome=outcome if turn_idx == len(history) - 1 else "ongoing",
                    suspicion_level=suspicion_level,
                    alliance_info=alliance_info
                )
                
                entries.append(entry)
                
        except Exception as e:
            print(f"Error processing game log {game_log.get('game_id', 'unknown')}: {e}")
            
        return entries

    def _extract_player_role(self, game_log: Dict[str, Any]) -> str:
        """Extract player role from game log"""
        history = game_log.get('history', [])
        if not history:
            return 'unknown'
            
        first_observation = history[0].get('observation', '')
        
        for role, pattern in self.role_patterns.items():
            if re.search(pattern, first_observation):
                return role
                
        return 'unknown'

    def _determine_outcome(self, game_log: Dict[str, Any]) -> str:
        """Determine game outcome from log"""
        filename = game_log.get('game_id', '')
        
        if 'win' in filename.lower():
            return 'win'
        elif 'loss' in filename.lower():
            return 'loss'
        elif 'error' in filename.lower():
            return 'error'
        else:
            return 'unknown'

    def _extract_game_phase(self, observation: str) -> str:
        """Extract current game phase"""
        for phase, pattern in self.phase_patterns.items():
            if re.search(pattern, observation, re.IGNORECASE):
                return phase
        return 'unknown'

    def _extract_reasoning_context(self, observation: str, action: str) -> str:
        """Extract reasoning context from observation and action"""
        # Look for reasoning patterns in the action
        reasoning_patterns = [
            r'I (think|believe|suspect|analyze)',
            r'Based on (.*?),',
            r'The evidence suggests',
            r'My strategy is',
            r'I need to'
        ]
        
        reasoning_context = []
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, action, re.IGNORECASE)
            reasoning_context.extend(matches)
        
        return ' | '.join(reasoning_context) if reasoning_context else ''

    def _extract_strategic_thinking(self, action: str) -> str:
        """Extract strategic thinking patterns"""
        # Look for explicit strategic reasoning
        strategic_patterns = [
            r'(My strategy|I plan to|I should|I will)',
            r'(To avoid|To minimize|To maximize)',
            r'(If .* then|When .* I will)',
            r'(This will help|This allows me to)'
        ]
        
        strategic_thoughts = []
        for pattern in strategic_patterns:
            matches = re.findall(pattern, action, re.IGNORECASE)
            strategic_thoughts.extend(matches)
        
        return ' | '.join(strategic_thoughts) if strategic_thoughts else ''

    def _extract_suspicion_level(self, observation: str) -> str:
        """Extract suspicion level indicators"""
        high_suspicion = ['highly suspicious', 'very suspicious', 'definitely mafia']
        medium_suspicion = ['suspicious', 'might be mafia', 'unclear']
        low_suspicion = ['not suspicious', 'probably village', 'trust']
        
        observation_lower = observation.lower()
        
        if any(phrase in observation_lower for phrase in high_suspicion):
            return 'high'
        elif any(phrase in observation_lower for phrase in medium_suspicion):
            return 'medium'
        elif any(phrase in observation_lower for phrase in low_suspicion):
            return 'low'
        else:
            return 'unknown'

    def _extract_alliance_info(self, observation: str, action: str) -> str:
        """Extract alliance formation information"""
        alliance_patterns = [
            r'ally with (\w+)',
            r'trust (\w+)',
            r'work with (\w+)',
            r'coordinate with (\w+)'
        ]
        
        combined_text = observation + ' ' + action
        alliances = []
        
        for pattern in alliance_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            alliances.extend(matches)
        
        return ' | '.join(alliances) if alliances else ''

    def create_training_examples(self, entries: List[GameLogEntry]) -> List[TrainingExample]:
        """Convert game entries to training examples"""
        training_examples = []
        
        for entry in entries:
            # Create SFT examples
            sft_examples = self._create_sft_examples(entry)
            training_examples.extend(sft_examples)
            
            # Create DPO examples (for preference learning)
            dpo_examples = self._create_dpo_examples(entry, entries)
            training_examples.extend(dpo_examples)
        
        return training_examples

    def _create_sft_examples(self, entry: GameLogEntry) -> List[TrainingExample]:
        """Create supervised fine-tuning examples"""
        examples = []
        
        # Skip low-quality entries
        if not self._is_high_quality_entry(entry):
            return examples
        
        # Role-specific instruction templates
        role_instructions = {
            'detective': "You are playing Mafia as a Detective. Your goal is to identify Mafia members through investigation and logical reasoning.",
            'doctor': "You are playing Mafia as a Doctor. Your goal is to protect village members while maintaining your cover.",
            'villager': "You are playing Mafia as a Villager. Your goal is to identify and eliminate Mafia members through discussion and voting.",
            'mafia': "You are playing Mafia as a Mafia member. Your goal is to eliminate village members while avoiding detection."
        }
        
        base_instruction = role_instructions.get(entry.player_role, "You are playing a social deduction game called Mafia.")
        
        # Create main gameplay example
        main_example = TrainingExample(
            instruction=f"{base_instruction} Analyze the current situation and decide on your action.",
            input_context=self._format_input_context(entry),
            target_response=entry.action_taken,
            reasoning_chain=self._format_reasoning_chain(entry),
            quality_score=self._calculate_quality_score(entry),
            example_type="sft"
        )
        examples.append(main_example)
        
        # Create strategic reasoning example
        if entry.strategic_thinking:
            strategy_example = TrainingExample(
                instruction=f"As a {entry.player_role} in Mafia, explain your strategic thinking for this situation.",
                input_context=entry.observation,
                target_response=entry.strategic_thinking,
                reasoning_chain=f"Role: {entry.player_role} | Phase: {entry.game_phase} | Strategic considerations",
                quality_score=self._calculate_quality_score(entry),
                example_type="sft"
            )
            examples.append(strategy_example)
        
        return examples

    def _create_dpo_examples(self, entry: GameLogEntry, all_entries: List[GameLogEntry]) -> List[TrainingExample]:
        """Create DPO (preference learning) examples"""
        examples = []
        
        # Find similar situations with different outcomes
        similar_entries = self._find_similar_entries(entry, all_entries)
        
        for similar_entry in similar_entries:
            if self._should_create_preference_pair(entry, similar_entry):
                # Create preference pair
                chosen_example, rejected_example = self._create_preference_pair(entry, similar_entry)
                examples.extend([chosen_example, rejected_example])
        
        return examples

    def _format_input_context(self, entry: GameLogEntry) -> str:
        """Format input context for training"""
        context_parts = [
            f"Game Phase: {entry.game_phase}",
            f"Turn: {entry.turn_number}",
            "",
            "Observation:",
            entry.observation
        ]
        
        if entry.reasoning_context:
            context_parts.extend(["", "Previous Reasoning:", entry.reasoning_context])
        
        return "\n".join(context_parts)

    def _format_reasoning_chain(self, entry: GameLogEntry) -> str:
        """Format reasoning chain for training"""
        reasoning_parts = [
            f"Role: {entry.player_role}",
            f"Phase: {entry.game_phase}",
            f"Suspicion Level: {entry.suspicion_level}"
        ]
        
        if entry.strategic_thinking:
            reasoning_parts.append(f"Strategic Thinking: {entry.strategic_thinking}")
        
        if entry.alliance_info:
            reasoning_parts.append(f"Alliance Considerations: {entry.alliance_info}")
        
        return " | ".join(reasoning_parts)

    def _is_high_quality_entry(self, entry: GameLogEntry) -> bool:
        """Assess if entry meets quality thresholds"""
        # Check minimum reasoning length
        if len(entry.action_taken) < self.min_reasoning_length:
            return False
        
        # Check for strategic indicators
        strategic_count = sum(1 for indicator in self.strategic_indicators 
                            if indicator.lower() in entry.action_taken.lower())
        
        if strategic_count < self.min_strategic_indicators:
            return False
        
        # Check for game phase validity
        if entry.game_phase == 'unknown':
            return False
        
        return True

    def _calculate_quality_score(self, entry: GameLogEntry) -> float:
        """Calculate quality score for training example"""
        score = 0.5  # Base score
        
        # Strategic thinking bonus
        strategic_count = sum(1 for indicator in self.strategic_indicators 
                            if indicator.lower() in entry.action_taken.lower())
        score += min(0.3, strategic_count * 0.05)
        
        # Length bonus (up to reasonable limit)
        length_bonus = min(0.2, len(entry.action_taken) / 1000)
        score += length_bonus
        
        # Successful outcome bonus
        if entry.outcome == 'win':
            score += 0.1
        
        # Role-specific reasoning bonus
        role_keywords = {
            'detective': ['investigate', 'evidence', 'suspect'],
            'doctor': ['protect', 'heal', 'save'],
            'villager': ['vote', 'discuss', 'analyze'],
            'mafia': ['deflect', 'mislead', 'survive']
        }
        
        role_words = role_keywords.get(entry.player_role, [])
        role_bonus = sum(0.02 for word in role_words 
                        if word in entry.action_taken.lower())
        score += min(0.1, role_bonus)
        
        return min(1.0, score)

    def _find_similar_entries(self, entry: GameLogEntry, all_entries: List[GameLogEntry]) -> List[GameLogEntry]:
        """Find similar entries for preference learning"""
        similar = []
        
        for other_entry in all_entries:
            if (other_entry.player_role == entry.player_role and 
                other_entry.game_phase == entry.game_phase and
                other_entry.game_id != entry.game_id):
                similar.append(other_entry)
        
        return similar[:5]  # Limit to 5 similar entries

    def _should_create_preference_pair(self, entry1: GameLogEntry, entry2: GameLogEntry) -> bool:
        """Determine if two entries should form a preference pair"""
        # Different outcomes suggest preference
        if entry1.outcome != entry2.outcome and entry1.outcome in ['win', 'loss'] and entry2.outcome in ['win', 'loss']:
            return True
        
        # Different quality scores suggest preference
        score1 = self._calculate_quality_score(entry1)
        score2 = self._calculate_quality_score(entry2)
        
        if abs(score1 - score2) > 0.2:
            return True
        
        return False

    def _create_preference_pair(self, entry1: GameLogEntry, entry2: GameLogEntry) -> Tuple[TrainingExample, TrainingExample]:
        """Create chosen/rejected pair for DPO"""
        score1 = self._calculate_quality_score(entry1)
        score2 = self._calculate_quality_score(entry2)
        
        if score1 > score2 or entry1.outcome == 'win':
            chosen, rejected = entry1, entry2
        else:
            chosen, rejected = entry2, entry1
        
        chosen_example = TrainingExample(
            instruction=f"As a {chosen.player_role}, respond to this game situation optimally.",
            input_context=self._format_input_context(chosen),
            target_response=chosen.action_taken,
            reasoning_chain=self._format_reasoning_chain(chosen),
            quality_score=self._calculate_quality_score(chosen),
            example_type="dpo_chosen"
        )
        
        rejected_example = TrainingExample(
            instruction=f"As a {rejected.player_role}, respond to this game situation optimally.",
            input_context=self._format_input_context(rejected),
            target_response=rejected.action_taken,
            reasoning_chain=self._format_reasoning_chain(rejected),
            quality_score=self._calculate_quality_score(rejected),
            example_type="dpo_rejected"
        )
        
        return chosen_example, rejected_example

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
    memory=8192
)
def process_game_logs_batch(log_files: List[str]) -> Dict[str, Any]:
    """Process a batch of game log files on Modal"""
    processor = MafiaGameLogProcessor()
    
    all_entries = []
    all_training_examples = []
    
    stats = {
        'processed_games': 0,
        'total_entries': 0,
        'high_quality_entries': 0,
        'sft_examples': 0,
        'dpo_examples': 0,
        'role_distribution': {},
        'outcome_distribution': {}
    }
    
    for log_file in log_files:
        try:
            # Load game log
            with open(f"/data/raw_logs/{log_file}", 'r') as f:
                game_log = json.load(f)
            
            # Extract entries
            entries = processor.extract_game_entries(game_log)
            all_entries.extend(entries)
            
            # Create training examples
            training_examples = processor.create_training_examples(entries)
            all_training_examples.extend(training_examples)
            
            # Update statistics
            stats['processed_games'] += 1
            stats['total_entries'] += len(entries)
            
            for entry in entries:
                # Role distribution
                if entry.player_role not in stats['role_distribution']:
                    stats['role_distribution'][entry.player_role] = 0
                stats['role_distribution'][entry.player_role] += 1
                
                # Outcome distribution
                if entry.outcome not in stats['outcome_distribution']:
                    stats['outcome_distribution'][entry.outcome] = 0
                stats['outcome_distribution'][entry.outcome] += 1
                
                # Quality assessment
                if processor._is_high_quality_entry(entry):
                    stats['high_quality_entries'] += 1
            
            # Count example types
            for example in training_examples:
                if example.example_type == 'sft':
                    stats['sft_examples'] += 1
                elif 'dpo' in example.example_type:
                    stats['dpo_examples'] += 1
                    
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            continue
    
    # Save processed training data
    output_file = f"/data/processed/training_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    with open(output_file, 'w') as f:
        for example in all_training_examples:
            example_dict = {
                'instruction': example.instruction,
                'input': example.input_context,
                'output': example.target_response,
                'reasoning': example.reasoning_chain,
                'quality_score': example.quality_score,
                'type': example.example_type
            }
            f.write(json.dumps(example_dict) + '\n')
    
    return {
        'output_file': output_file,
        'statistics': stats,
        'total_examples': len(all_training_examples)
    }

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200
)
def create_training_dataset():
    """Main function to create complete training dataset"""
    
    # List all game log files
    log_dir = "/data/raw_logs"
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    
    print(f"Found {len(log_files)} game log files")
    
    # Process in batches for parallel processing
    batch_size = 10
    batches = [log_files[i:i+batch_size] for i in range(0, len(log_files), batch_size)]
    
    # Process batches in parallel
    batch_results = []
    for batch in batches:
        result = process_game_logs_batch.remote(batch)
        batch_results.append(result)
    
    # Collect results
    all_stats = {
        'total_processed_games': 0,
        'total_entries': 0,
        'total_high_quality_entries': 0,
        'total_sft_examples': 0,
        'total_dpo_examples': 0,
        'role_distribution': {},
        'outcome_distribution': {}
    }
    
    processed_files = []
    
    for result in batch_results:
        batch_stats = result['statistics']
        processed_files.append(result['output_file'])
        
        # Aggregate statistics
        all_stats['total_processed_games'] += batch_stats['processed_games']
        all_stats['total_entries'] += batch_stats['total_entries']
        all_stats['total_high_quality_entries'] += batch_stats['high_quality_entries']
        all_stats['total_sft_examples'] += batch_stats['sft_examples']
        all_stats['total_dpo_examples'] += batch_stats['dpo_examples']
        
        # Merge distributions
        for role, count in batch_stats['role_distribution'].items():
            if role not in all_stats['role_distribution']:
                all_stats['role_distribution'][role] = 0
            all_stats['role_distribution'][role] += count
        
        for outcome, count in batch_stats['outcome_distribution'].items():
            if outcome not in all_stats['outcome_distribution']:
                all_stats['outcome_distribution'][outcome] = 0
            all_stats['outcome_distribution'][outcome] += count
    
    # Combine all processed files into final datasets
    sft_file = "/data/final/mafia_sft_dataset.jsonl"
    dpo_file = "/data/final/mafia_dpo_dataset.jsonl"
    
    # Separate SFT and DPO examples
    with open(sft_file, 'w') as sft_f, open(dpo_file, 'w') as dpo_f:
        for processed_file in processed_files:
            with open(processed_file, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    if example['type'] == 'sft':
                        sft_f.write(line)
                    elif 'dpo' in example['type']:
                        dpo_f.write(line)
    
    # Save final statistics
    with open("/data/final/dataset_statistics.json", 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"Dataset creation complete!")
    print(f"SFT examples: {all_stats['total_sft_examples']}")
    print(f"DPO examples: {all_stats['total_dpo_examples']}")
    print(f"Total high-quality entries: {all_stats['total_high_quality_entries']}")
    
    return {
        'sft_dataset': sft_file,
        'dpo_dataset': dpo_file,
        'statistics': all_stats
    }

@app.local_entrypoint()
def main():
    """Local entrypoint for running preprocessing"""
    # Upload raw game logs to Modal volume first
    # This should be done separately before running preprocessing
    
    print("Starting Mafia training dataset creation...")
    result = create_training_dataset.remote()
    
    print("Preprocessing completed successfully!")
    print(f"SFT dataset: {result['sft_dataset']}")
    print(f"DPO dataset: {result['dpo_dataset']}")
    
    return result

if __name__ == "__main__":
    main()