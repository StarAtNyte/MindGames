# SecretMafia Agent - Stage 2

**Team:** MG25-3162A7F500

**Model:** Qwen/Qwen3-8B (ZeroR-SecretMafia-Efficient-v4)

**Track:** Social Detection (Small Model Track)

---

### Installation (No Extra Steps)

Recommended python version: 3.12

```bash
# Create an environment (optional)
conda create -n zeror python=3.12
conda activate zeror

# Clone repository
git clone https://github.com/StarAtNyte/MindGames.git
cd SecretMafia

# Install all dependencies (versions pinned in requirements.txt)
pip install -r requirements.txt

# That's it! No extra configuration needed.
```

### Run Instructions (Single Entry Point)

**Option 1: Local GPU (Recommended)**
```bash
cd src
python local_play.py
```

**Option 2: Modal API (Fallback)**
```bash
cd src
python online_play.py
```

**No CLI args or config files needed** - All settings are in the script files with sensible defaults.

In case you encounter any problem install a specific cuda version: 

```bash

pip install torch --index-url https://download.pytorch.org/whl/cu121


---

## 🚀 Quick Start (Recommended: Local Inference)

Run the agent **entirely locally** on your GPU with zero API costs and 11x faster inference:

```bash
# Install dependencies
pip install -r requirements.txt

# Run local agent
cd src
python local_play.py
```

**First run:** ~30-60s model loading
**Subsequent turns:** ~1-3s inference (vs 20-30s on Modal)
**Requirements:** RTX 4090 or better (24GB+ VRAM)

---

## Overview

MindGames Challenge 2025, Stage 2 submission for the Social Detection track (Small Model). This agent uses Qwen3-8B with Theory of Mind reasoning for strategic social deduction gameplay.

## Model Description

- **Base Model:** Qwen/Qwen3-8B (8 billion parameters)
- **Architecture:**
  - Theory of Mind engine for multi-level belief modeling
  - Two-step reasoning: internal analysis followed by public action generation
  - Phase-aware decision making (discussion, voting, night actions)
  - Pure LLM decision making with no heuristic post-processing
  - All strategic reasoning performed by the language model
- **Edge Case Reliability:** 100% pass rate on all stress tests
- **Performance:** 11.7x faster with local GPU vs Modal endpoint

## Installation

### Prerequisites

**For Local Inference (Recommended):**
- **Python:** 3.10 or higher
- **GPU:** RTX 4090 (24GB VRAM) or better
- **CUDA:** 11.8 or 12.1+
- **Storage:** 20GB free space (for model)
- **OS:** Linux (recommended), macOS, or Windows with WSL

**For Modal Endpoint (Alternative):**
- **Python:** 3.10 or higher
- **RAM:** 4GB minimum
- **Internet:** Required for API access

### Step 1: Clone the Repository

```bash
git clone https://github.com/StarAtNyte/MindGames.git
cd SecretMafia
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch>=2.3.0` - PyTorch for GPU inference
- `transformers>=4.40.0` - Hugging Face models
- `accelerate>=0.30.0` - Fast model loading
- `bitsandbytes>=0.43.0` - Quantization support
- `textarena==1.0.0` - Game environment
- Other required packages

**For specific CUDA versions:**

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1+
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Verify GPU setup (for local inference)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify TextArena
python -c "import textarena; print('TextArena installed successfully')"
```

Expected GPU output:
```
CUDA available: True
GPU: NVIDIA RTX 4090
```

## Running Matches

### 🚀 Method 1: Local Inference (Recommended)

Run the agent entirely on your GPU with **11x faster inference** and zero API costs:

```bash
cd src
python local_play.py
```

**What happens:**
1. Loads Qwen3-8B locally (~30s first time, instant thereafter)
2. Connects to MindGames game servers
3. Plays games using local GPU inference (~1-3s per turn)
4. Saves logs to `game_logs_local/`

**Configuration:**

Edit `src/local_play.py`:

```python
MODEL_NAME = "Qwen/Qwen3-8B"  # Base model
USE_4BIT = True               # 4-bit quantization (RTX 4090)
                              # False for A100/H100 (better quality)
team_hash = "MG25-3162A7F500-LOCAL"
```

**Performance:**
- First load: 30-60s (model loading)
- Subsequent turns: 1-3s (vs 20-30s on Modal)
- Total speed improvement: **11.7x faster**
- Edge case reliability: **100% pass rate**

### Method 2: Modal Endpoint (Alternative)

Use the cloud-hosted Modal endpoint:

#### Step 1: Warm Up Endpoint

```bash
python warmup_endpoint.py
```

#### Step 2: Run Game

```bash
cd src
python online_play.py
```

**What happens:**
1. Connects to Modal endpoint at `https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run`
2. Joins online matchmaking
3. Plays games using Modal API (~20-30s per turn with cold starts)
4. Saves logs to `game_logs/`

**Configuration:**

Edit `src/online_play.py`:

```python
MODEL_NAME = "ZeroR-SecretMafia-Efficient-v4"
team_hash = "MG25-3162A7F500"
modal_endpoint_url = "https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run"
```


## Performance Comparison

Real edge case test results:

| Metric | Local GPU (RTX 4090) | Modal Endpoint |
|--------|---------------------|----------------|
| **Test Duration** | 104.6s (~1.7 min) | 1228.2s (~20 min) |
| **Speed** | **11.7x faster** | Baseline |
| **Edge Case Pass Rate** | 100% (10/10) | 100% (10/10) |
| **Cold Start** | 17s (first load only) | 30-60s (every cold start) |
| **Subsequent Turns** | 1-3s | 20-30s |
| **Cost** | $0 (free) | Modal compute costs |
| **Reliability** | ✅ Perfect | ✅ Perfect |

**Recommendation:** Use local inference for development and production. Modal is available as a fallback.

## Testing & Validation

### Edge Case Tests

Both local and Modal agents passed **100% of edge case tests**:

```bash
# Test local agent
python test_local_edge_cases.py

# Test Modal endpoint
python test_all_edge_cases.py

# Compare both
python compare_edge_cases.py
```

**Tests include:**
- Invalid/corrupted observations
- Timeout scenarios (complex reasoning)
- Consecutive game stability (5 games)
- Long game stability (20+ turns)
- Response format validation
- Variable player counts (3-15 players)
- Role reveal protection
- Invalid move recovery
- Memory overflow protection
- Meta-commentary detection

**Result:** Both agents achieve **10/10 (100%) pass rate**

### Manual Testing

Test the local agent without game server:

```bash
cd src
python -c "
from utils.local_llm_agent import LocalStreamlinedMafiaAgent
agent = LocalStreamlinedMafiaAgent(model_name='Qwen/Qwen3-8B', load_in_4bit=True)
response = agent('Test observation')
print(response)
"
```

Test the Modal endpoint:

```bash
cd src
python -c "
from utils.streamlined_mafia_agent import StreamlinedMafiaAgent
agent = StreamlinedMafiaAgent('https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run')
response = agent('Test observation')
print(response)
"
```

## Model Hosting

### Option 1: Local Inference (Recommended)

See "Method 1: Local Inference" in the Running Matches section above.

### Option 2: Modal Endpoint

Our model is hosted on Modal at:
```
https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run
```

**Endpoints:**
- `POST /generate` - Generate agent response
- `GET /health` - Health check
- `GET /` - API information

The hosted model uses Qwen3-8B with 8-bit quantization on an A100 GPU.

**Health Check:**

```bash
curl https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run/health
```

### Deploy Your Own Modal Endpoint (Optional)

If you want to deploy on your own Modal account:

1. **Install Modal dependencies:**
```bash
pip install modal==0.64.0
```

2. **Set up Modal authentication:**
```bash
modal token new
```

3. **Deploy the model:**
```bash
cd src/utils
modal deploy modal_secretmafia.py
```

4. **Update endpoint URL in `src/online_play.py`:**
```python
modal_endpoint_url = "<your-modal-endpoint-url>"
```

## Architecture

### Core Components

1. **StreamlinedMafiaAgent** (`src/utils/streamlined_mafia_agent.py`)
   - Main agent class that coordinates all decision-making
   - Two-step reasoning: internal analysis → public action
   - Phase detection (discussion/voting/night actions)
   - Game state tracking and memory management

2. **AdvancedToMEngine** (`src/utils/theory_of_mind.py`)
   - First-order beliefs: What I believe about each player
   - Second-order beliefs: What I think others believe
   - Behavioral pattern tracking and analysis
   - Alliance and trust network detection
   - Strategic insights generation

3. **Memory System**
   - **SimpleGameMemory** (`src/utils/streamlined_mafia_agent.py`)
     - Investigation results tracking
     - Role claims from players
     - Voting pattern analysis
     - Bounded discussion history (last 5 turns for memory efficiency)
     - LLM observation summarization cache

4. **Local Inference** (Recommended)
   - **LocalStreamlinedMafiaAgent** (`src/utils/local_llm_agent.py`)
     - Loads Qwen3-8B directly on GPU using transformers
     - 4-bit/8-bit quantization support via bitsandbytes
     - Modal-compatible API (drop-in replacement)
     - Same two-step reasoning as Modal agent
     - 11.7x faster inference on local GPU

5. **Modal Inference** (Fallback)
   - **StreamlinedMafiaAgent** (`src/utils/streamlined_mafia_agent.py`)
     - HTTP client for Modal endpoint
     - Automatic retry and timeout handling
     - Response parsing and validation
   - **Modal Endpoint** (`src/utils/modal_secretmafia.py`)
     - FastAPI server running Qwen3-8B
     - Phase-aware prompt engineering
     - GPU inference with 8-bit quantization

### Decision Flow

```
Observation → Phase Detection → Game State Update → ToM Analysis →
Prompt Generation → LLM Inference (Local GPU or Modal) →
Response Extraction → Action
```

## Game Logs

Games are automatically logged with comprehensive telemetry:

**Local inference:** `game_logs_local/game_YYYYMMDD_HHMMSS_<role>_<outcome>.json`
**Modal endpoint:** `game_logs/game_YYYYMMDD_HHMMSS_<role>_<outcome>.json`

Example: `game_20251029_143000_mafia_win.json`

Each log contains:
- Full observation and action history
- Turn-by-turn analysis with timestamps
- Performance metrics (processing time, memory usage)
- Agent memory state (investigation results, role claims, voting patterns)
- Theory of Mind belief updates
- Error logs and recovery attempts

## Why Local Inference?

Based on real testing data from edge case test suite:

| Benefit | Local GPU (RTX 4090) | Modal Endpoint |
|---------|---------------------|----------------|
| **Speed** | **11.7x faster** (104s) | Baseline (1228s) |
| **Cost** | $0 (free) | Modal compute costs |
| **Reliability** | 100% pass rate | 100% pass rate |
| **Cold Start** | 17s (once) | 30-60s (every time) |
| **Turn Time** | 1-3s | 20-30s |
| **Control** | Full | Limited |
| **Offline** | ✅ Yes | ❌ No |

**Recommendation:** Use local inference for development and production. Modal is available as fallback.

## Contributing

See existing code structure for examples. Key files:
- `src/utils/local_llm_agent.py` - Local inference implementation
- `src/utils/streamlined_mafia_agent.py` - Modal client + shared logic
- `src/utils/theory_of_mind.py` - ToM reasoning engine
- `test_local_edge_cases.py` - Comprehensive test suite

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **MindGames Challenge 2025** - Competition organizers
- **Qwen Team** - Base model (Qwen3-8B)
- **Modal Labs** - Cloud inference platform
- **Hugging Face** - Model hosting and transformers library
