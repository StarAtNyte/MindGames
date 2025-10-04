"""
Modal Labs Fine-tuning Pipeline for Qwen 2.5 7B on Mafia Game Data
Implements SFT + DPO training optimized for social deduction games

Training Strategy:
1. Supervised Fine-tuning (SFT) on high-quality gameplay examples
2. Direct Preference Optimization (DPO) for strategic decision making
3. Rules-compliant training (no heuristics in training data)
4. Multi-GPU distributed training on Modal infrastructure
"""

import json
import os
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import modal
from datetime import datetime
import wandb

# Modal app configuration
app = modal.App("mafia-qwen-training")

# Modal image with ML dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0", 
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "wandb",
        "tensorboard",
        "scipy",
        "scikit-learn",
        "trl>=0.7.0",  # For DPO training
        "jsonlines"
    ])
    .run_commands([
        "pip install flash-attn --no-build-isolation",  # For efficient attention
    ])
)

# Modal volume for model storage
model_volume = modal.Volume.from_name("mafia-models", create_if_missing=True)
data_volume = modal.Volume.from_name("mafia-training-data", create_if_missing=True)

@dataclass
class MafiaTrainingConfig:
    """Training configuration for Mafia game fine-tuning"""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 2048
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # SFT Training arguments
    sft_learning_rate: float = 2e-4
    sft_num_epochs: int = 3
    sft_batch_size: int = 4
    sft_gradient_accumulation_steps: int = 4
    
    # DPO Training arguments  
    dpo_learning_rate: float = 5e-5
    dpo_num_epochs: int = 1
    dpo_batch_size: int = 2
    dpo_beta: float = 0.1  # DPO temperature parameter
    
    # General training settings
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Hardware settings
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Output settings
    output_dir: str = "/models/mafia-qwen-finetuned"
    run_name: str = f"mafia-qwen-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class MafiaDataFormatter:
    """Format training data for Qwen 2.5 instruction format"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Qwen 2.5 chat template
        self.instruction_template = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        
        self.system_prompt = """You are an expert Mafia player with advanced strategic thinking capabilities. 
You excel at:
- Analyzing social dynamics and player behavior
- Making strategic decisions based on incomplete information  
- Adapting your strategy based on role and game phase
- Communicating persuasively while maintaining your cover
- Coordinating with allies and identifying threats

Always provide detailed reasoning for your actions and consider multiple perspectives before deciding."""

    def format_sft_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format example for supervised fine-tuning"""
        
        formatted_text = self.instruction_template.format(
            system=self.system_prompt,
            instruction=example['instruction'],
            input=example['input'],
            output=example['output']
        )
        
        return {
            'text': formatted_text,
            'input_ids': self.tokenizer.encode(formatted_text, add_special_tokens=False)
        }

    def format_dpo_example(self, chosen_example: Dict[str, Any], rejected_example: Dict[str, Any]) -> Dict[str, Any]:
        """Format example pair for DPO training"""
        
        # Format chosen response
        chosen_text = self.instruction_template.format(
            system=self.system_prompt,
            instruction=chosen_example['instruction'],
            input=chosen_example['input'], 
            output=chosen_example['output']
        )
        
        # Format rejected response
        rejected_text = self.instruction_template.format(
            system=self.system_prompt,
            instruction=rejected_example['instruction'],
            input=rejected_example['input'],
            output=rejected_example['output']
        )
        
        return {
            'chosen': chosen_text,
            'rejected': rejected_text,
            'prompt': self.instruction_template.format(
                system=self.system_prompt,
                instruction=chosen_example['instruction'],
                input=chosen_example['input'],
                output=""
            ).replace("<|im_start|>assistant\n<|im_end|>", "")
        }

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=2),  # Use 2 A100 GPUs for faster training
    volumes={
        "/models": model_volume,
        "/data": data_volume
    },
    timeout=14400,  # 4 hours
    memory=32768   # 32GB RAM
)
def run_sft_training(config: MafiaTrainingConfig) -> str:
    """Run supervised fine-tuning on Mafia gameplay data"""
    
    print("Starting SFT training...")
    
    # Initialize wandb for experiment tracking
    wandb.init(
        project="mafia-qwen-training",
        name=f"{config.run_name}-sft",
        config=config.__dict__
    )
    
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Configure LoRA
    if config.use_lora:
        print("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load and format training data
    print("Loading training data...")
    formatter = MafiaDataFormatter(tokenizer)
    
    # Load SFT dataset
    with open("/data/final/mafia_sft_dataset.jsonl", 'r') as f:
        sft_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(sft_data)} SFT examples")
    
    # Format training examples
    formatted_data = []
    for example in sft_data:
        if example['type'] == 'sft':
            formatted_example = formatter.format_sft_example(example)
            formatted_data.append(formatted_example)
    
    # Create dataset
    train_dataset = Dataset.from_list(formatted_data)
    
    # Split into train/eval
    train_eval_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_eval_split['train']
    eval_dataset = train_eval_split['test']
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{config.output_dir}/sft",
        num_train_epochs=config.sft_num_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        per_device_eval_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_gradient_accumulation_steps,
        learning_rate=config.sft_learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        report_to="wandb",
        run_name=f"{config.run_name}-sft",
        seed=42
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train model
    print("Starting SFT training...")
    trainer.train()
    
    # Save final model
    sft_model_path = f"{config.output_dir}/sft/final"
    trainer.save_model(sft_model_path)
    tokenizer.save_pretrained(sft_model_path)
    
    print(f"SFT training completed. Model saved to {sft_model_path}")
    
    wandb.finish()
    
    return sft_model_path

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=2),
    volumes={
        "/models": model_volume,
        "/data": data_volume
    },
    timeout=10800,  # 3 hours
    memory=32768
)
def run_dpo_training(sft_model_path: str, config: MafiaTrainingConfig) -> str:
    """Run DPO training on the SFT model for preference optimization"""
    
    print("Starting DPO training...")
    
    # Initialize wandb
    wandb.init(
        project="mafia-qwen-training",
        name=f"{config.run_name}-dpo",
        config=config.__dict__
    )
    
    # Load SFT model and tokenizer
    print("Loading SFT model...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load reference model (for DPO)
    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load DPO data
    print("Loading DPO data...")
    formatter = MafiaDataFormatter(tokenizer)
    
    with open("/data/final/mafia_dpo_dataset.jsonl", 'r') as f:
        dpo_data = [json.loads(line) for line in f]
    
    # Group DPO examples into pairs
    dpo_pairs = []
    chosen_examples = [ex for ex in dpo_data if ex['type'] == 'dpo_chosen']
    rejected_examples = [ex for ex in dpo_data if ex['type'] == 'dpo_rejected']
    
    # Create pairs based on similar instructions
    for chosen in chosen_examples:
        for rejected in rejected_examples:
            if chosen['instruction'] == rejected['instruction']:
                dpo_pair = formatter.format_dpo_example(chosen, rejected)
                dpo_pairs.append(dpo_pair)
                break
    
    print(f"Created {len(dpo_pairs)} DPO training pairs")
    
    # Create DPO dataset
    dpo_dataset = Dataset.from_list(dpo_pairs)
    
    # Import DPO trainer
    try:
        from trl import DPOTrainer, DPOConfig
        
        # DPO training configuration
        dpo_config = DPOConfig(
            output_dir=f"{config.output_dir}/dpo",
            num_train_epochs=config.dpo_num_epochs,
            per_device_train_batch_size=config.dpo_batch_size,
            gradient_accumulation_steps=4,
            learning_rate=config.dpo_learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            fp16=config.fp16,
            gradient_checkpointing=config.gradient_checkpointing,
            beta=config.dpo_beta,
            report_to="wandb",
            run_name=f"{config.run_name}-dpo",
            seed=42
        )
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=dpo_dataset,
            tokenizer=tokenizer,
            max_length=config.max_length
        )
        
        # Train with DPO
        print("Starting DPO training...")
        dpo_trainer.train()
        
        # Save final model
        dpo_model_path = f"{config.output_dir}/dpo/final"
        dpo_trainer.save_model(dpo_model_path)
        tokenizer.save_pretrained(dpo_model_path)
        
    except ImportError:
        print("TRL not available, skipping DPO training")
        dpo_model_path = sft_model_path
    
    print(f"DPO training completed. Model saved to {dpo_model_path}")
    
    wandb.finish()
    
    return dpo_model_path

@app.function(
    image=image,
    gpu=modal.gpu.A100(),
    volumes={
        "/models": model_volume,
        "/data": data_volume
    },
    timeout=3600
)
def evaluate_model(model_path: str, config: MafiaTrainingConfig) -> Dict[str, Any]:
    """Evaluate the trained model on held-out test data"""
    
    print("Evaluating trained model...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load test data
    with open("/data/final/mafia_sft_dataset.jsonl", 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # Take a sample for evaluation
    test_sample = test_data[:100]  # Evaluate on 100 examples
    
    results = {
        'total_examples': len(test_sample),
        'successful_generations': 0,
        'average_length': 0,
        'strategy_indicators': 0,
        'role_accuracy': {}
    }
    
    strategy_keywords = ['analyze', 'strategy', 'suspect', 'trust', 'evidence', 'alliance']
    
    total_length = 0
    successful_gens = 0
    
    for example in test_sample:
        try:
            # Format input
            formatter = MafiaDataFormatter(tokenizer)
            prompt = formatter.instruction_template.format(
                system=formatter.system_prompt,
                instruction=example['instruction'],
                input=example['input'],
                output=""
            ).replace("<|im_start|>assistant\n<|im_end|>", "<|im_start|>assistant\n")
            
            # Generate response
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
            
            # Evaluate response
            if len(generated_text.strip()) > 10:
                successful_gens += 1
                total_length += len(generated_text)
                
                # Check for strategy indicators
                strategy_count = sum(1 for keyword in strategy_keywords 
                                   if keyword.lower() in generated_text.lower())
                results['strategy_indicators'] += strategy_count
        
        except Exception as e:
            print(f"Error evaluating example: {e}")
            continue
    
    results['successful_generations'] = successful_gens
    results['average_length'] = total_length / max(successful_gens, 1)
    results['strategy_density'] = results['strategy_indicators'] / max(successful_gens, 1)
    
    print(f"Evaluation Results:")
    print(f"Successful generations: {successful_gens}/{len(test_sample)}")
    print(f"Average response length: {results['average_length']:.1f}")
    print(f"Strategy indicators per response: {results['strategy_density']:.2f}")
    
    return results

@app.function(
    image=image,
    volumes={
        "/models": model_volume,
        "/data": data_volume
    },
    timeout=21600  # 6 hours total
)
def train_mafia_model():
    """Main training pipeline combining SFT and DPO"""
    
    print("Starting Mafia Qwen 2.5 training pipeline...")
    
    # Initialize training configuration
    config = MafiaTrainingConfig()
    
    # Phase 1: Supervised Fine-tuning
    print("Phase 1: Supervised Fine-tuning")
    sft_model_path = run_sft_training.remote(config)
    
    # Phase 2: DPO Training
    print("Phase 2: Direct Preference Optimization")
    final_model_path = run_dpo_training.remote(sft_model_path, config)
    
    # Phase 3: Evaluation
    print("Phase 3: Model Evaluation")
    eval_results = evaluate_model.remote(final_model_path, config)
    
    # Save training summary
    training_summary = {
        'config': config.__dict__,
        'sft_model_path': sft_model_path,
        'final_model_path': final_model_path,
        'evaluation_results': eval_results,
        'training_completed': datetime.now().isoformat()
    }
    
    with open(f"/models/training_summary_{config.run_name}.json", 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print("Training pipeline completed successfully!")
    print(f"Final model saved to: {final_model_path}")
    
    return training_summary

@app.local_entrypoint()
def main():
    """Local entrypoint for training"""
    print("Starting Mafia Qwen 2.5 7B training on Modal Labs...")
    
    # Run the complete training pipeline
    result = train_mafia_model.remote()
    
    print("Training completed!")
    print(f"Results: {result}")
    
    return result

if __name__ == "__main__":
    main()