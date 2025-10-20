#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B on Mafia game data using Modal Labs with Unsloth LoRA
"""

import modal
import json
import os
from datetime import datetime

# Create Modal app
app = modal.App("qwen3-mafia-finetune")

# Create volume for training data
volume = modal.Volume.from_name("mafia-data", create_if_missing=True)

# Define training image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git"])  # Install git first
    .pip_install([
        "torch>=2.1.0",
        "torchvision>=0.16.0",  # Required by unsloth
        "transformers>=4.36.0", 
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "wandb",
    ])
    .pip_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")  # Install unsloth separately after git
    .env({"WANDB_DISABLED": "true"})  # Disable wandb for now
)

@app.function(
    image=training_image,
    gpu="H100",  # H100 80GB - no compromises!
    memory=192000,  # Max memory for H100
    timeout=86400,  # 24 hours
    volumes={"/data": volume},
)
def train_qwen3_lora(
    train_file: str = "mafia_training_data_train.json",
    val_file: str = "mafia_training_data_val.json", 
    model_name: str = "Qwen/Qwen3-8B",
    output_dir: str = "/data/qwen3-mafia-lora",
    max_seq_length: int = 2048,
    batch_size: int = 8,  # Larger batch size with H100 80GB
    gradient_accumulation_steps: int = 2,  # Less accumulation needed
    learning_rate: float = 1e-4,  # Lower LR for longer training
    num_epochs: int = 8,  # More epochs for small dataset
    lora_r: int = 32,     # Higher rank for more capacity
    lora_alpha: int = 64,  # 2x rank for stronger adaptation
    lora_dropout: float = 0.05,
):
    """Fine-tune Qwen3-8B with LoRA on Mafia game data"""
    
    # Check if model is already trained
    import os
    merged_path = "/data/qwen3-mafia-merged"
    if os.path.exists(merged_path):
        print(f"✅ Fine-tuned model already exists at {merged_path}")
        print("Skipping training. Delete the directory to retrain.")
        return {
            "status": "skipped", 
            "message": "Model already exists",
            "model_path": merged_path,
            "training_duration": "0:00:00",
            "num_train_examples": 0,
            "num_val_examples": 0
        }
    
    print("🚀 Starting Qwen3-8B LoRA Fine-tuning")
    print("=" * 50)
    
    # Import libraries
    import torch
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import Dataset
    import json
    
    # Load training data
    print(f"📚 Loading training data...")
    with open(f"/data/{train_file}", 'r') as f:
        train_data = json.load(f)
    
    with open(f"/data/{val_file}", 'r') as f:
        val_data = json.load(f)
    
    print(f"   Training examples: {len(train_data)}")
    print(f"   Validation examples: {len(val_data)}")
    
    # Load model with Unsloth - NO QUANTIZATION on H100!
    print(f"🔧 Loading {model_name} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,  # Use bfloat16 for best performance on H100
        load_in_4bit=False,  # NO quantization - we have H100 80GB!
    )
    
    # Add LoRA adapters
    print(f"⚙️ Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Long context support
        random_state=42,
    )
    
    # Format data for training
    def format_conversation(example):
        """Format conversation for Qwen3 chat template"""
        conversations = example['conversations']
        
        # Convert 'value' to 'content' for Qwen3 chat template compatibility
        formatted_conversations = []
        for msg in conversations:
            formatted_msg = {
                "role": msg["from"] if msg["from"] != "assistant" else "assistant",
                "content": msg["value"]
            }
            # Map 'from' field to 'role' field for chat template
            if msg["from"] == "system":
                formatted_msg["role"] = "system"
            elif msg["from"] == "human":
                formatted_msg["role"] = "user"
            elif msg["from"] == "assistant":
                formatted_msg["role"] = "assistant"
            formatted_conversations.append(formatted_msg)
        
        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            formatted_conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": formatted}
    
    # Create datasets
    print(f"📋 Formatting datasets...")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    train_dataset = train_dataset.map(format_conversation)
    val_dataset = val_dataset.map(format_conversation)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb
        remove_unused_columns=False,
    )
    
    # Create trainer
    print(f"🎯 Setting up SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False,  # Disable packing for chat format
    )
    
    # Train model
    print(f"🚂 Starting training...")
    start_time = datetime.now()
    
    trainer.train()
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print(f"✅ Training completed in {training_duration}")
    
    # Save model
    print(f"💾 Saving model...")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "training_duration": str(training_duration),
        "num_train_examples": len(train_data),
        "num_val_examples": len(val_data),
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "max_seq_length": max_seq_length,
        },
        "final_eval_loss": float(trainer.state.log_history[-1].get("eval_loss", 0)),
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"🎉 Fine-tuning complete!")
    print(f"📁 Model saved to: {output_dir}/final")
    
    return training_info

@app.function(
    image=training_image,
    gpu="H100",  # H100 for merging too
    memory=128000,  # More memory for merging
    timeout=86400,
    volumes={"/data": volume},
)
def merge_and_save_model(
    lora_path: str = "/data/qwen3-mafia-lora",
    base_model: str = "Qwen/Qwen3-8B",
    output_path: str = "/data/qwen3-mafia-merged"
):
    """Merge LoRA adapters with base model and save"""
    
    print("🔗 Merging LoRA adapters with base model...")
    
    from unsloth import FastLanguageModel
    import torch
    
    # Check for the correct checkpoint path
    import os
    if os.path.exists(lora_path):
        print(f"📁 LoRA directory contents: {os.listdir(lora_path)}")
        # Look for checkpoint directories or use the path directly
        checkpoint_dirs = [d for d in os.listdir(lora_path) if d.startswith('checkpoint-')]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
            lora_path = os.path.join(lora_path, checkpoint_dirs[-1])
            print(f"📍 Using checkpoint: {lora_path}")
    
    # Load model with LoRA - full precision on H100
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,  # Use bfloat16 for H100
        load_in_4bit=False,  # NO quantization on H100
    )
    
    # Merge and save
    model = FastLanguageModel.for_inference(model)
    model.save_pretrained_merged(
        output_path,
        tokenizer,
        save_method="merged_bf16",  # Save as bfloat16 for H100
    )
    
    print(f"✅ Merged model saved to: {output_path}")
    
    return output_path

@app.function(
    image=training_image,
    gpu="H100",  # H100 for evaluation too
    memory=64000,
    volumes={"/data": volume},
    timeout=86400,
)
def evaluate_model(
    model_path: str = "/data/qwen3-mafia-merged",
    test_examples: int = 10
):
    """Quick evaluation of fine-tuned model"""
    
    print("🧪 Evaluating fine-tuned model...")
    
    from unsloth import FastLanguageModel
    import torch
    import json
    
    # Load merged model - full precision on H100
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,  # bfloat16 on H100
        load_in_4bit=False,  # NO quantization
    )
    model = FastLanguageModel.for_inference(model)
    
    # Load validation data
    with open("/data/mafia_training_data_val.json", 'r') as f:
        val_data = json.load(f)
    
    # Test on a few examples
    results = []
    for i, example in enumerate(val_data[:test_examples]):
        conversations = example['conversations']
        
        # Get input (without assistant response)
        test_conversation = conversations[:-1]
        expected_output = conversations[-1]['value']
        
        # Generate response
        inputs = tokenizer.apply_chat_template(
            test_conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        ).strip()
        
        results.append({
            "input": test_conversation[-1]['value'][:100] + "...",
            "expected": expected_output,
            "generated": generated,
            "role": example['metadata']['role'],
            "phase": example['metadata']['phase'],
        })
        
        print(f"Example {i+1}:")
        print(f"  Role: {example['metadata']['role']}")
        print(f"  Phase: {example['metadata']['phase']}")
        print(f"  Expected: {expected_output}")
        print(f"  Generated: {generated}")
        print(f"  Match: {'✅' if expected_output == generated else '❌'}")
        print()
    
    return results

@app.local_entrypoint()
def main():
    """Main training pipeline"""
    print("🚀 Qwen3-8B Mafia Fine-tuning Pipeline")
    print("=" * 60)
    
    print("📋 Steps:")
    print("1. Train LoRA adapters")
    print("2. Merge with base model") 
    print("3. Evaluate performance")
    print("4. Save final model")
    
    # Step 1: Train LoRA
    print("\n🎯 Step 1: Training LoRA adapters...")
    training_info = train_qwen3_lora.remote()
    print(f"Training info: {training_info}")
    
    # Step 2: Merge model
    print("\n🔗 Step 2: Merging LoRA with base model...")
    merged_path = merge_and_save_model.remote()
    print(f"Merged model path: {merged_path}")
    
    # Step 3: Evaluate
    print("\n🧪 Step 3: Evaluating model...")
    eval_results = evaluate_model.remote(merged_path)
    
    print("\n✅ Fine-tuning pipeline complete!")
    print(f"📁 Final model location: {merged_path}")
    
    return {
        "training_info": training_info,
        "merged_path": merged_path,
        "eval_results": eval_results
    }

if __name__ == "__main__":
    # For local testing
    print("Run with: modal run modal_finetune_qwen3.py")