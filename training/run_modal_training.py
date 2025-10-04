"""
Complete Modal Labs Training Orchestration for Mafia Qwen 2.5 7B
Coordinates data upload, preprocessing, and training pipeline

Usage:
    python run_modal_training.py --upload --preprocess --train
    python run_modal_training.py --train-only  # Skip upload/preprocessing
    python run_modal_training.py --eval-only   # Only evaluate existing model
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import modal

# Import our Modal apps
from upload_to_modal import app as upload_app
from modal_preprocessing import app as preprocessing_app  
from modal_training import app as training_app

def run_complete_pipeline(
    game_logs_dir: str = "src/game_logs",
    skip_upload: bool = False,
    skip_preprocessing: bool = False,
    skip_training: bool = False,
    eval_only: bool = False
):
    """Run the complete training pipeline"""
    
    print("🎮 Mafia Qwen 2.5 7B Training Pipeline")
    print("=" * 50)
    
    pipeline_start = time.time()
    results = {}
    
    # Step 1: Data Upload
    if not skip_upload and not eval_only:
        print("\n📤 Step 1: Uploading game logs to Modal...")
        try:
            with upload_app.run():
                upload_result = upload_app.main.remote(game_logs_dir)
                results['upload'] = upload_result
                print(f"✅ Upload completed: {upload_result['upload_stats']['successful_uploads']} files")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return results
    else:
        print("\n⏭️  Step 1: Skipping upload (using existing data)")
    
    # Step 2: Data Preprocessing  
    if not skip_preprocessing and not eval_only:
        print("\n🔄 Step 2: Preprocessing game logs...")
        try:
            with preprocessing_app.run():
                preprocessing_result = preprocessing_app.create_training_dataset.remote()
                results['preprocessing'] = preprocessing_result
                print(f"✅ Preprocessing completed:")
                print(f"   SFT examples: {preprocessing_result['statistics']['total_sft_examples']}")
                print(f"   DPO examples: {preprocessing_result['statistics']['total_dpo_examples']}")
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return results
    else:
        print("\n⏭️  Step 2: Skipping preprocessing (using existing data)")
    
    # Step 3: Model Training
    if not skip_training and not eval_only:
        print("\n🚀 Step 3: Training Qwen 2.5 7B model...")
        try:
            with training_app.run():
                training_result = training_app.train_mafia_model.remote()
                results['training'] = training_result
                print(f"✅ Training completed!")
                print(f"   Final model: {training_result['final_model_path']}")
                print(f"   Evaluation: {training_result['evaluation_results']}")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return results
    elif eval_only:
        print("\n📊 Step 3: Evaluating existing model...")
        try:
            # Assuming we have a trained model path
            model_path = "/models/mafia-qwen-finetuned/dpo/final"
            with training_app.run():
                from modal_training import MafiaTrainingConfig, evaluate_model
                config = MafiaTrainingConfig()
                eval_result = evaluate_model.remote(model_path, config)
                results['evaluation'] = eval_result
                print(f"✅ Evaluation completed: {eval_result}")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return results
    else:
        print("\n⏭️  Step 3: Skipping training")
    
    # Summary
    pipeline_duration = time.time() - pipeline_start
    
    print(f"\n🎉 Pipeline completed in {pipeline_duration/60:.1f} minutes")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"training_results_{timestamp}.json"
    
    pipeline_summary = {
        'timestamp': timestamp,
        'duration_minutes': pipeline_duration / 60,
        'steps_completed': list(results.keys()),
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(pipeline_summary, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")
    
    return pipeline_summary

def run_inference_test(model_path: str = None):
    """Test the trained model with sample Mafia scenarios"""
    
    print("\n🧪 Running inference test...")
    
    # Sample test scenarios
    test_scenarios = [
        {
            "role": "detective",
            "phase": "discussion",
            "observation": "It's Day 2. Player 3 was eliminated (Villager). Players alive: 0, 1, 2, 4, 5. You investigated Player 2 last night - they are MAFIA.",
            "instruction": "You are a Detective who just discovered a Mafia member. Decide how to reveal this information strategically."
        },
        {
            "role": "mafia", 
            "phase": "discussion",
            "observation": "It's Day 3. Players alive: 0, 1, 3, 5. Player 1 is acting very suspicious and has been accusing everyone. You need to deflect suspicion.",
            "instruction": "You are Mafia and need to deflect suspicion while eliminating threats. Plan your response."
        },
        {
            "role": "doctor",
            "phase": "vote", 
            "observation": "Vote Phase: Players are discussing who to eliminate. The votes are split between Player 2 and Player 4. You believe Player 2 is innocent.",
            "instruction": "As a Doctor, decide how to vote and whether to reveal your role to save an innocent player."
        }
    ]
    
    if not model_path:
        print("No model path provided for inference test")
        return
    
    # This would require implementing an inference function in the training module
    print("Inference testing would require the trained model to be accessible")
    print("Test scenarios prepared:")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['role'].title()} in {scenario['phase']} phase")
    
    return test_scenarios

def estimate_training_costs():
    """Estimate Modal Labs training costs"""
    
    print("\n💰 Training Cost Estimation")
    print("-" * 30)
    
    # Modal A100 pricing (approximate)
    a100_cost_per_hour = 4.0  # $4/hour per A100
    
    estimates = {
        "Data Upload": {
            "duration_minutes": 30,
            "cost_usd": 0.50
        },
        "Preprocessing": {
            "duration_hours": 1,
            "cost_usd": 2.00
        },
        "SFT Training": {
            "duration_hours": 3,
            "gpus": 2,
            "cost_usd": 3 * 2 * a100_cost_per_hour
        },
        "DPO Training": {
            "duration_hours": 2, 
            "gpus": 2,
            "cost_usd": 2 * 2 * a100_cost_per_hour
        },
        "Evaluation": {
            "duration_minutes": 30,
            "gpus": 1,
            "cost_usd": 0.5 * a100_cost_per_hour
        }
    }
    
    total_cost = sum(step["cost_usd"] for step in estimates.values())
    
    for step, details in estimates.items():
        print(f"{step:15}: ${details['cost_usd']:6.2f}")
    
    print("-" * 30)
    print(f"{'Total Estimated':15}: ${total_cost:6.2f}")
    print(f"{'+ Modal overhead':15}: ${total_cost * 0.1:6.2f}")
    print(f"{'Grand Total':15}: ${total_cost * 1.1:6.2f}")
    
    return estimates

def main():
    parser = argparse.ArgumentParser(description="Modal Labs Mafia Training Pipeline")
    parser.add_argument("--game-logs-dir", default="src/game_logs", 
                       help="Directory containing game log JSON files")
    parser.add_argument("--upload", action="store_true",
                       help="Upload game logs to Modal")
    parser.add_argument("--preprocess", action="store_true", 
                       help="Preprocess game logs for training")
    parser.add_argument("--train", action="store_true",
                       help="Train the model")
    parser.add_argument("--all", action="store_true",
                       help="Run complete pipeline (upload + preprocess + train)")
    parser.add_argument("--train-only", action="store_true",
                       help="Skip upload/preprocessing, only train")
    parser.add_argument("--eval-only", action="store_true", 
                       help="Only evaluate existing model")
    parser.add_argument("--estimate-costs", action="store_true",
                       help="Show estimated training costs")
    parser.add_argument("--test-inference", 
                       help="Test inference with trained model (provide model path)")
    
    args = parser.parse_args()
    
    # Show cost estimates
    if args.estimate_costs:
        estimate_training_costs()
        return
    
    # Test inference
    if args.test_inference:
        run_inference_test(args.test_inference)
        return
    
    # Determine what steps to run
    if args.all:
        skip_upload = False
        skip_preprocessing = False  
        skip_training = False
        eval_only = False
    elif args.train_only:
        skip_upload = True
        skip_preprocessing = True
        skip_training = False
        eval_only = False
    elif args.eval_only:
        skip_upload = True
        skip_preprocessing = True
        skip_training = True
        eval_only = True
    else:
        skip_upload = not args.upload
        skip_preprocessing = not args.preprocess
        skip_training = not args.train
        eval_only = False
    
    # Validate game logs directory
    if not skip_upload:
        game_logs_path = Path(args.game_logs_dir)
        if not game_logs_path.exists():
            print(f"❌ Game logs directory not found: {args.game_logs_dir}")
            print("Please ensure the directory exists and contains JSON game log files")
            return
        
        json_files = list(game_logs_path.glob("*.json"))
        if not json_files:
            print(f"❌ No JSON files found in: {args.game_logs_dir}")
            return
        
        print(f"📁 Found {len(json_files)} game log files in {args.game_logs_dir}")
    
    # Run the pipeline
    try:
        results = run_complete_pipeline(
            game_logs_dir=args.game_logs_dir,
            skip_upload=skip_upload,
            skip_preprocessing=skip_preprocessing, 
            skip_training=skip_training,
            eval_only=eval_only
        )
        
        print("\n🎉 Pipeline execution completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()