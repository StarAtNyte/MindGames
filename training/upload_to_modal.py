"""
Upload Mafia Game Logs to Modal Labs for Training
Transfers local game logs to Modal volume for distributed processing
"""

import os
import json
import modal
from pathlib import Path
from typing import List, Dict, Any
import shutil

# Modal app for data upload
app = modal.App("mafia-data-upload")

# Modal volume for storing training data
data_volume = modal.Volume.from_name("mafia-training-data", create_if_missing=True)

@app.function(
    volumes={"/data": data_volume},
    timeout=3600
)
def upload_game_logs(game_logs_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Upload game logs to Modal volume"""
    
    # Ensure directories exist
    os.makedirs("/data/raw_logs", exist_ok=True)
    os.makedirs("/data/processed", exist_ok=True)
    os.makedirs("/data/final", exist_ok=True)
    
    upload_stats = {
        'total_files': len(game_logs_data),
        'successful_uploads': 0,
        'failed_uploads': 0,
        'total_size_bytes': 0
    }
    
    for i, game_log in enumerate(game_logs_data):
        try:
            # Generate filename based on game log content
            game_id = game_log.get('game_id', f'game_{i:04d}')
            filename = f"{game_id}.json"
            
            # Save to Modal volume
            file_path = f"/data/raw_logs/{filename}"
            with open(file_path, 'w') as f:
                json.dump(game_log, f, indent=2)
            
            # Update stats
            file_size = os.path.getsize(file_path)
            upload_stats['total_size_bytes'] += file_size
            upload_stats['successful_uploads'] += 1
            
            print(f"Uploaded {filename} ({file_size} bytes)")
            
        except Exception as e:
            print(f"Failed to upload game log {i}: {e}")
            upload_stats['failed_uploads'] += 1
    
    print(f"Upload completed: {upload_stats['successful_uploads']}/{upload_stats['total_files']} files")
    print(f"Total size: {upload_stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
    
    return upload_stats

def load_local_game_logs(game_logs_dir: str) -> List[Dict[str, Any]]:
    """Load game logs from local directory"""
    
    game_logs = []
    game_logs_path = Path(game_logs_dir)
    
    if not game_logs_path.exists():
        print(f"Game logs directory not found: {game_logs_dir}")
        return game_logs
    
    print(f"Loading game logs from: {game_logs_dir}")
    
    # Find all JSON files
    json_files = list(game_logs_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                game_log = json.load(f)
            
            # Add metadata
            game_log['game_id'] = json_file.stem
            game_log['source_file'] = str(json_file)
            
            game_logs.append(game_log)
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(game_logs)} game logs")
    return game_logs

def analyze_game_logs(game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze game logs for quality and distribution"""
    
    analysis = {
        'total_games': len(game_logs),
        'role_distribution': {},
        'outcome_distribution': {},
        'valid_games': 0,
        'games_with_history': 0,
        'average_turns': 0,
        'quality_issues': []
    }
    
    total_turns = 0
    
    for game_log in game_logs:
        # Check if game has history
        history = game_log.get('history', [])
        if history:
            analysis['games_with_history'] += 1
            total_turns += len(history)
        
        # Extract role from filename or game data
        game_id = game_log.get('game_id', '')
        
        # Extract role from filename pattern
        role = 'unknown'
        for role_name in ['DETECTIVE', 'DOCTOR', 'VILLAGER', 'MAFIA']:
            if role_name in game_id.upper():
                role = role_name.lower()
                break
        
        if role not in analysis['role_distribution']:
            analysis['role_distribution'][role] = 0
        analysis['role_distribution'][role] += 1
        
        # Extract outcome from filename
        outcome = 'unknown'
        if 'win' in game_id.lower():
            outcome = 'win'
        elif 'loss' in game_id.lower():
            outcome = 'loss'
        elif 'error' in game_id.lower():
            outcome = 'error'
        
        if outcome not in analysis['outcome_distribution']:
            analysis['outcome_distribution'][outcome] = 0
        analysis['outcome_distribution'][outcome] += 1
        
        # Quality checks
        if history and len(history) > 2:
            analysis['valid_games'] += 1
        else:
            analysis['quality_issues'].append(f"{game_id}: insufficient history")
    
    if analysis['games_with_history'] > 0:
        analysis['average_turns'] = total_turns / analysis['games_with_history']
    
    return analysis

@app.local_entrypoint()
def main(game_logs_dir: str = "src/game_logs"):
    """Main function to upload game logs to Modal"""
    
    print("=== Mafia Game Logs Upload to Modal Labs ===")
    
    # Load local game logs
    print("Step 1: Loading local game logs...")
    game_logs = load_local_game_logs(game_logs_dir)
    
    if not game_logs:
        print("No game logs found. Please check the directory path.")
        return
    
    # Analyze game logs
    print("\nStep 2: Analyzing game logs...")
    analysis = analyze_game_logs(game_logs)
    
    print(f"\nGame Logs Analysis:")
    print(f"Total games: {analysis['total_games']}")
    print(f"Valid games: {analysis['valid_games']}")
    print(f"Games with history: {analysis['games_with_history']}")
    print(f"Average turns per game: {analysis['average_turns']:.1f}")
    
    print(f"\nRole distribution:")
    for role, count in analysis['role_distribution'].items():
        print(f"  {role}: {count}")
    
    print(f"\nOutcome distribution:")
    for outcome, count in analysis['outcome_distribution'].items():
        print(f"  {outcome}: {count}")
    
    if analysis['quality_issues']:
        print(f"\nQuality issues found: {len(analysis['quality_issues'])}")
        for issue in analysis['quality_issues'][:5]:  # Show first 5
            print(f"  {issue}")
        if len(analysis['quality_issues']) > 5:
            print(f"  ... and {len(analysis['quality_issues']) - 5} more")
    
    # Upload to Modal
    print("\nStep 3: Uploading to Modal Labs...")
    upload_stats = upload_game_logs.remote(game_logs)
    
    print(f"\nUpload completed!")
    print(f"Successfully uploaded: {upload_stats['successful_uploads']}")
    print(f"Failed uploads: {upload_stats['failed_uploads']}")
    print(f"Total size: {upload_stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
    
    # Save analysis report
    with open("game_logs_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to: game_logs_analysis.json")
    print("Ready for preprocessing and training!")
    
    return {
        'analysis': analysis,
        'upload_stats': upload_stats
    }

if __name__ == "__main__":
    # Run with default game logs directory
    main("src/game_logs")