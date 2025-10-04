# Mafia Qwen 2.5 7B Training on Modal Labs

Complete guide for training Qwen 2.5 7B on Mafia game logs using Modal Labs infrastructure.

## Overview

This training pipeline implements:
- **Supervised Fine-tuning (SFT)** on high-quality Mafia gameplay examples
- **Direct Preference Optimization (DPO)** for strategic decision making
- **Rules-compliant training** (no heuristics in data or model)
- **Distributed processing** on Modal Labs A100 GPUs

## Files Structure

```
SecretMafia/
├── modal_preprocessing.py     # Data preprocessing pipeline
├── modal_training.py         # SFT + DPO training pipeline  
├── upload_to_modal.py        # Upload game logs to Modal
├── run_modal_training.py     # Complete orchestration script
├── modal_requirements.txt    # Python dependencies
└── MODAL_TRAINING_GUIDE.md   # This guide
```

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate
3. **Weights & Biases**: Account for experiment tracking (optional)
4. **Game Logs**: JSON files in `src/game_logs/`

## Setup

### 1. Install Dependencies

```bash
pip install -r modal_requirements.txt
```

### 2. Configure Modal

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token set

# Test connection
modal --help
```

### 3. Set Environment Variables

Create `.env` file (optional for W&B tracking):
```bash
WANDB_API_KEY=your_wandb_key_here
```

## Usage

### Quick Start (Complete Pipeline)

```bash
# Run everything: upload + preprocess + train
python run_modal_training.py --all

# Or step by step
python run_modal_training.py --upload --preprocess --train
```

### Step-by-Step Execution

#### 1. Upload Game Logs
```bash
python run_modal_training.py --upload --game-logs-dir src/game_logs
```

#### 2. Preprocess Data
```bash
python run_modal_training.py --preprocess
```

#### 3. Train Model
```bash
python run_modal_training.py --train
```

### Training Only (Skip Upload/Preprocessing)

```bash
python run_modal_training.py --train-only
```

### Evaluation Only

```bash
python run_modal_training.py --eval-only
```

### Cost Estimation

```bash
python run_modal_training.py --estimate-costs
```

## Training Configuration

### Model Settings
- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **LoRA**: r=16, alpha=32, dropout=0.1
- **Max Length**: 2048 tokens
- **Hardware**: 2x A100 GPUs

### Training Parameters
- **SFT**: 3 epochs, 2e-4 learning rate
- **DPO**: 1 epoch, 5e-5 learning rate, beta=0.1
- **Batch Size**: 4 (SFT), 2 (DPO)
- **Gradient Accumulation**: 4 steps

### Data Processing
- **Quality Filtering**: Min 50 chars, 2+ strategic indicators
- **Role Distribution**: Detective, Doctor, Villager, Mafia
- **Example Types**: SFT (direct supervision), DPO (preference pairs)

## Training Pipeline Details

### Phase 1: Data Upload
- Transfers local JSON game logs to Modal volume
- Validates file format and content
- Generates data quality report

### Phase 2: Preprocessing
- **Parallel Processing**: Batched across multiple Modal functions
- **Quality Assessment**: Filters low-quality examples
- **Format Conversion**: Creates SFT and DPO training examples
- **Strategic Analysis**: Extracts reasoning patterns and strategic thinking

### Phase 3: Training

#### SFT (Supervised Fine-tuning)
- Trains on high-quality gameplay examples
- Uses Qwen 2.5 instruction format
- Implements LoRA for efficient training
- Tracks metrics with W&B

#### DPO (Direct Preference Optimization)  
- Optimizes strategic decision making
- Uses win/loss outcomes for preferences
- Implements preference learning objective
- Fine-tunes from SFT checkpoint

### Phase 4: Evaluation
- Tests on held-out examples
- Measures response quality and strategy indicators
- Generates evaluation report

## Expected Outputs

### Training Artifacts
```
/models/mafia-qwen-finetuned/
├── sft/final/                 # SFT model checkpoint
├── dpo/final/                 # Final DPO model  
└── training_summary_*.json    # Training metrics
```

### Data Artifacts
```
/data/
├── raw_logs/                  # Original game logs
├── processed/                 # Intermediate processing
└── final/
    ├── mafia_sft_dataset.jsonl
    ├── mafia_dpo_dataset.jsonl  
    └── dataset_statistics.json
```

## Cost Estimation

| Component | Duration | GPUs | Est. Cost |
|-----------|----------|------|-----------|
| Upload | 30 min | 0 | $0.50 |
| Preprocessing | 1 hour | 0 | $2.00 |
| SFT Training | 3 hours | 2x A100 | $24.00 |
| DPO Training | 2 hours | 2x A100 | $16.00 |
| Evaluation | 30 min | 1x A100 | $2.00 |
| **Total** | **~7 hours** | | **~$45-50** |

*Costs may vary based on Modal pricing and resource availability*

## Training Data Format

### SFT Example
```json
{
  "instruction": "You are playing Mafia as a Detective. Analyze the situation and decide your action.",
  "input": "Game Phase: discussion\nTurn: 3\n\nObservation:\nIt's Day 3. Player 2 was eliminated (Villager)...",
  "output": "I need to carefully analyze the voting patterns from yesterday...",
  "reasoning": "Role: detective | Phase: discussion | Strategic thinking detected",
  "quality_score": 0.85,
  "type": "sft"
}
```

### DPO Example Pair
```json
{
  "chosen": "High-quality strategic response...",
  "rejected": "Lower-quality response...", 
  "prompt": "Game situation requiring strategic decision...",
  "type": "dpo_chosen"
}
```

## Monitoring and Debugging

### Weights & Biases Integration
- Real-time training metrics
- Loss curves and learning rates
- Model performance tracking
- Experiment comparison

### Modal Logs
```bash
# View function logs
modal logs list

# Stream live logs
modal logs follow <function-name>
```

### Debug Mode
Add debug prints in functions and view via Modal dashboard.

## Troubleshooting

### Common Issues

#### Upload Failures
- Check Modal authentication: `modal token list`
- Verify game logs directory exists and contains JSON files
- Ensure sufficient Modal credits

#### Training OOM (Out of Memory)
- Reduce batch size in `MafiaTrainingConfig`
- Enable gradient checkpointing
- Use smaller LoRA rank

#### Model Loading Issues
- Verify Qwen 2.5 model access
- Check HuggingFace authentication if using private models
- Ensure sufficient disk space in Modal volume

#### Performance Issues
- Monitor GPU utilization in Modal dashboard
- Check data loading bottlenecks
- Verify distributed training setup

### Log Analysis
```bash
# Check preprocessing logs
modal logs list --app mafia-training-preprocessing

# Check training logs  
modal logs list --app mafia-qwen-training
```

## Best Practices

### Data Quality
- Filter out error games and incomplete sessions
- Ensure balanced role and outcome distribution
- Maintain high reasoning quality thresholds

### Training Efficiency
- Use gradient accumulation for larger effective batch sizes
- Implement early stopping based on evaluation metrics
- Save checkpoints frequently for recovery

### Cost Optimization
- Use spot instances when available
- Batch preprocessing for parallel execution
- Monitor resource utilization and adjust accordingly

### Rules Compliance
- Ensure no heuristic patterns in training data
- Verify all decisions come from LLM reasoning
- Document training methodology for competition submission

## Next Steps

1. **Model Deployment**: Deploy trained model for online play
2. **Performance Analysis**: Compare against baseline agent
3. **Iterative Improvement**: Collect more high-quality training data
4. **Ablation Studies**: Test different training configurations

## Support

For issues with this training pipeline:
1. Check Modal documentation: [docs.modal.com](https://docs.modal.com)
2. Review training logs for specific error messages
3. Test individual components before running full pipeline
4. Monitor resource usage and costs during training

---

*This training pipeline is designed for the MindGames competition and follows all rules regarding LLM-based reasoning without heavy heuristics.*