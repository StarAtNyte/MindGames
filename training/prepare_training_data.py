#!/usr/bin/env python3
"""
Prepare training data from Mafia game logs for Qwen3-8B fine-tuning
Converts game logs to Alpaca format for LoRA training
"""

import json
import os
import re
from typing import List, Dict, Any
from datetime import datetime
import random

def detect_phase(observation: str) -> str:
    """Detect game phase from observation text"""
    obs_lower = observation.lower()
    
    # Night actions
    if any(phrase in obs_lower for phrase in [
        "night phase", "choose one player to investigate", 
        "choose one player to protect", "night has fallen",
        "mafia, agree on a victim"
    ]):
        return "night_action"
    
    # Voting phase  
    if any(phrase in obs_lower for phrase in [
        "voting phase", "submit one vote in format"
    ]):
        return "voting_action"
    
    # Discussion phase
    if any(phrase in obs_lower for phrase in [
        "day breaks", "discuss for", "what do you think"
    ]):
        return "discussion"
    
    return "unknown"

def is_valid_response(response: str, phase: str) -> bool:
    """Check if response format is valid for the phase"""
    if not response or len(response.strip()) == 0:
        return False
    
    # Action phases should have [X] format
    if phase in ["night_action", "voting_action"]:
        return bool(re.match(r'^\s*\[\d+\]\s*$', response.strip()))
    
    # Discussion should be natural language (not [X] format)
    elif phase == "discussion":
        action_format = bool(re.match(r'^\s*\[\d+\]\s*$', response.strip()))
        return not action_format and len(response.strip()) > 10
    
    return True

def extract_role_from_filename(filename: str) -> str:
    """Extract role from filename"""
    if "mafia" in filename.lower():
        return "Mafia"
    elif "detective" in filename.lower():
        return "Detective" 
    elif "doctor" in filename.lower():
        return "Doctor"
    elif "villager" in filename.lower():
        return "Villager"
    else:
        return "Unknown"

def parse_game_log(filepath: str) -> List[Dict]:
    """Parse a single game log file into training examples"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    
    # Skip error games
    if game_data.get('outcome') == 'error':
        return []
    
    # Extract basic info
    role = game_data.get('role', extract_role_from_filename(os.path.basename(filepath)))
    outcome = game_data.get('outcome', 'unknown')
    
    training_examples = []
    
    # Process observations and actions
    observations = game_data.get('observations', [])
    actions = game_data.get('actions', [])
    
    for obs, act in zip(observations, actions):
        obs_content = obs.get('content', '')
        act_content = act.get('action', '')
        
        if not obs_content or not act_content:
            continue
            
        phase = detect_phase(obs_content)
        
        # Skip if response format is invalid for phase
        if not is_valid_response(act_content, phase):
            continue
        
        # Advanced fine-tuning techniques complementing inference-time ToM/strategic reasoning
        if phase in ["night_action", "voting_action"]:
            # Self-Consistency: Train on multiple valid reasoning paths to same action
            system_prompt = f"""You are a {role} in a Mafia game. Multiple reasoning paths can lead to optimal actions.
Consider: threat assessment, information value, team coordination, timing.
Respond with exactly [number] format."""
        else:
            # Multi-step reasoning chain + preference learning from winning games
            if outcome == 'win':
                system_prompt = f"""You are a {role} in a Mafia game. This is a WINNING strategy example.
Break down your analysis: 1) Assess current situation 2) Identify key players 3) Plan communication strategy 4) Execute discussion.
Focus on information gathering, trust building, and logical reasoning that leads to victory."""
            else:
                system_prompt = f"""You are a {role} in a Mafia game. Learn from this game outcome.
Analyze: what information was missed, which communication strategies failed, how timing affected results.
Focus on adaptive reasoning and information synthesis."""
        
        training_example = {
            "conversations": [
                {
                    "from": "system",
                    "value": system_prompt
                },
                {
                    "from": "human", 
                    "value": obs_content
                },
                {
                    "from": "assistant",
                    "value": act_content
                }
            ],
            "metadata": {
                "role": role,
                "phase": phase,
                "outcome": outcome,
                "turn": obs.get('turn', 0),
                "source_file": os.path.basename(filepath)
            }
        }
        
        training_examples.append(training_example)
    
    return training_examples

def create_contrastive_pairs(examples: List[Dict]) -> List[Dict]:
    """Create contrastive learning pairs from win/loss examples"""
    contrastive_examples = []
    
    # Group by role and phase for contrastive pairing
    role_phase_groups = {}
    for ex in examples:
        role = ex['metadata']['role']
        phase = ex['metadata']['phase']
        key = f"{role}_{phase}"
        
        if key not in role_phase_groups:
            role_phase_groups[key] = {'win': [], 'loss': []}
        
        outcome = ex['metadata']['outcome']
        if outcome == 'win':
            role_phase_groups[key]['win'].append(ex)
        else:
            role_phase_groups[key]['loss'].append(ex)
    
    # Create contrastive pairs
    for key, group in role_phase_groups.items():
        wins = group['win']
        losses = group['loss']
        
        # Create preference pairs: winning strategy > losing strategy
        for win_ex in wins[:50]:  # Limit to avoid too many pairs
            if losses:
                loss_ex = random.choice(losses)
                # Create DPO-style preference pair
                contrastive_pair = {
                    "conversations": [
                        win_ex['conversations'][0],  # Same system prompt
                        win_ex['conversations'][1],  # Same human input
                        {
                            "from": "assistant",
                            "value": win_ex['conversations'][2]['value']  # Preferred (winning) response
                        }
                    ],
                    "rejected": loss_ex['conversations'][2]['value'],  # Rejected (losing) response
                    "metadata": {
                        **win_ex['metadata'],
                        "training_type": "preference_pair",
                        "preferred_outcome": "win",
                        "rejected_outcome": "loss"
                    }
                }
                contrastive_examples.append(contrastive_pair)
    
    print(f"📊 Created {len(contrastive_examples)} contrastive preference pairs")
    return contrastive_examples

def filter_training_data(examples: List[Dict]) -> List[Dict]:
    """Filter and balance training data with advanced techniques"""
    # Group by role and phase
    role_phase_groups = {}
    for ex in examples:
        role = ex['metadata']['role']
        phase = ex['metadata']['phase']
        outcome = ex['metadata']['outcome']
        key = f"{role}_{phase}_{outcome}"
        
        if key not in role_phase_groups:
            role_phase_groups[key] = []
        role_phase_groups[key].append(ex)
    
    # Print distribution
    print("\n📊 Data Distribution:")
    for key, examples_list in role_phase_groups.items():
        print(f"   {key}: {len(examples_list)} examples")
    
    # Create contrastive learning pairs
    contrastive_pairs = create_contrastive_pairs(examples)
    
    # Balance standard examples - emphasize winning patterns
    max_per_group = 80  # Reduced to make room for contrastive pairs
    balanced_examples = []
    
    for key, examples_list in role_phase_groups.items():
        if 'win' in key:  # Prioritize winning examples
            sample_size = min(len(examples_list), int(max_per_group * 1.2))  # 20% more winning examples
        else:
            sample_size = min(len(examples_list), max_per_group)
        
        if len(examples_list) > sample_size:
            sampled = random.sample(examples_list, sample_size)
            balanced_examples.extend(sampled)
        else:
            balanced_examples.extend(examples_list)
    
    # Add contrastive pairs
    balanced_examples.extend(contrastive_pairs)
    
    # Add self-consistency examples (multiple reasoning paths to same action)
    consistency_examples = create_self_consistency_examples(examples)
    balanced_examples.extend(consistency_examples)
    
    print(f"\n✅ Enhanced dataset: {len(balanced_examples)} examples")
    print(f"   📝 Standard examples: {len(balanced_examples) - len(contrastive_pairs) - len(consistency_examples)}")
    print(f"   🎯 Contrastive pairs: {len(contrastive_pairs)}")
    print(f"   🔄 Self-consistency: {len(consistency_examples)}")
    
    return balanced_examples

def create_self_consistency_examples(examples: List[Dict]) -> List[Dict]:
    """Create self-consistency training examples with multiple reasoning paths"""
    consistency_examples = []
    
    # Find examples with same action but different contexts (different reasoning paths)
    action_groups = {}
    for ex in examples:
        if ex['metadata']['phase'] in ['night_action', 'voting_action']:
            action = ex['conversations'][2]['value']  # The action taken
            role = ex['metadata']['role']
            key = f"{role}_{action}"
            
            if key not in action_groups:
                action_groups[key] = []
            action_groups[key].append(ex)
    
    # Create consistency examples from groups with multiple instances
    for key, group in action_groups.items():
        if len(group) >= 3:  # Need at least 3 examples for variety
            # Sample different contexts leading to same action
            sampled = random.sample(group, min(3, len(group)))
            
            for i, ex in enumerate(sampled):
                consistency_ex = {
                    "conversations": [
                        {
                            "from": "system",
                            "value": f"{ex['conversations'][0]['value']} Multiple reasoning paths can lead to this optimal action."
                        },
                        ex['conversations'][1],  # Same observation
                        ex['conversations'][2]   # Same action
                    ],
                    "metadata": {
                        **ex['metadata'],
                        "training_type": "self_consistency",
                        "consistency_group": key,
                        "variation": i
                    }
                }
                consistency_examples.append(consistency_ex)
    
    print(f"🔄 Created {len(consistency_examples)} self-consistency examples")
    return consistency_examples

def create_training_dataset(game_logs_dir: str, output_file: str):
    """Create training dataset from all game logs"""
    print(f"🔄 Processing game logs from: {game_logs_dir}")
    
    all_examples = []
    processed_files = 0
    error_files = 0
    
    # Process all JSON files
    for filename in os.listdir(game_logs_dir):
        if not filename.endswith('.json'):
            continue
            
        # Skip obvious error files
        if 'error' in filename.lower() and 'unknown_error' in filename.lower():
            error_files += 1
            continue
        
        filepath = os.path.join(game_logs_dir, filename)
        examples = parse_game_log(filepath)
        
        if examples:
            all_examples.extend(examples)
            processed_files += 1
            
            if processed_files % 50 == 0:
                print(f"   Processed {processed_files} files, {len(all_examples)} examples so far...")
    
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Processed files: {processed_files}")
    print(f"   ❌ Skipped error files: {error_files}")
    print(f"   📝 Raw examples: {len(all_examples)}")
    
    # Filter and balance
    filtered_examples = filter_training_data(all_examples)
    
    # Split train/validation
    random.shuffle(filtered_examples)
    split_idx = int(len(filtered_examples) * 0.9)
    
    train_data = filtered_examples[:split_idx]
    val_data = filtered_examples[split_idx:]
    
    # Save datasets
    train_file = output_file.replace('.json', '_train.json')
    val_file = output_file.replace('.json', '_val.json')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Dataset created:")
    print(f"   📚 Training: {len(train_data)} examples → {train_file}")
    print(f"   📊 Validation: {len(val_data)} examples → {val_file}")
    
    # Show sample
    print(f"\n📋 Sample training example:")
    sample = train_data[0]
    print(f"   Role: {sample['metadata']['role']}")
    print(f"   Phase: {sample['metadata']['phase']}")
    print(f"   Input: {sample['conversations'][1]['value'][:100]}...")
    print(f"   Output: {sample['conversations'][2]['value']}")
    
    return train_file, val_file

def main():
    """Main function"""
    print("🚀 Preparing Mafia Game Training Data for Qwen3-8B Fine-tuning")
    print("=" * 60)
    
    # Configuration
    game_logs_dir = "../src/game_logs"
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mafia_training_data.json")
    
    # Check if game logs exist
    if not os.path.exists(game_logs_dir):
        print(f"❌ Game logs directory not found: {game_logs_dir}")
        return
    
    # Create training dataset
    train_file, val_file = create_training_dataset(game_logs_dir, output_file)
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Upload to Modal: modal volume put mafia-data {train_file} {val_file}")
    print(f"   2. Run training: modal run modal_finetune_qwen3.py")
    print(f"   3. Deploy fine-tuned model to production endpoint")
    
    return train_file, val_file

if __name__ == "__main__":
    main()