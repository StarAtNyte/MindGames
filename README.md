# SecretMafia Agent - Stage 2 Submission

**Team:** MG25-3162A7F500
**Model:** ZeroR-SecretMafia-Efficient-v4
**Track:** Social Detection (Small Model Track)

## Overview

This is our submission for the MindGames Challenge 2025, Stage 2 (Small Model Track). Our agent plays Secret Mafia using a fine-tuned Qwen3-8B model with advanced Theory of Mind reasoning and strategic decision-making capabilities.

## Model Description

- **Base Model:** Qwen/Qwen3-8B (8 billion parameters)
- **Architecture:** Streamlined agent with:
  - Advanced Theory of Mind (ToM) engine for belief modeling
  - Strategic bidding system for turn-taking optimization
  - Two-step reasoning (internal analysis → public action)
  - Phase-aware decision making (discussion/voting/night actions)
  - Rules-compliant pure LLM decision making (no heuristics)

## Installation

### Prerequisites

- **Python:** 3.12 or higher
- **OS:** Linux, macOS, or Windows with WSL
- **RAM:** 4GB minimum (for client only, not running the model locally)
- **Internet:** Required for accessing the Modal-hosted model endpoint

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd SecretMafia
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `textarena==1.0.0` - Game environment
- `requests==2.31.0` - HTTP client for model endpoint
- `numpy==1.26.4` - Data processing

### Step 3: Verify Installation

```bash
python3 --version  # Should be 3.12+
python3 -c "import textarena; print('textarena installed successfully')"
```

## Running Matches

### Method 1

You can also run directly from the source directory:

```bash
cd src
python3 online_play.py
```

### Alternative Method

Use the single entry point script:

```bash
python3 run_match.py
```

This will:
1. Connect to the Modal endpoint at `https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run`
2. Join online matchmaking
3. Play one game
4. Save game log to `src/game_logs/` directory

Both methods are equivalent and play one game per run.

### Configuration

The main configuration is in `src/online_play.py`:

```python
MODEL_NAME = "ZeroR-SecretMafia-Efficient-v4"
MODEL_DESCRIPTION = "Streamlined agent with Theory of Mind, Strategic Bidding, Rules Compliant"
team_hash = "MG25-3162A7F500"
modal_endpoint_url = "https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run"
```


## Model Hosting

### Hosted Model

Our model is hosted on Modal at:
```
https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run
```

**Endpoints:**
- `POST /generate` - Generate agent response
- `GET /health` - Health check
- `GET /` - API information

The hosted model uses Qwen3-8B with 8-bit quantization on an A100 GPU.

### Local Model Deployment (Optional)

If you want to deploy the model locally or on your own Modal account:

1. **Install Modal dependencies:**
```bash
pip install modal==0.64.0 transformers==4.51.0 torch==2.0.0
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

4. **Update the endpoint URL in `src/online_play.py`:**
```python
modal_endpoint_url = "<your-modal-endpoint-url>"
```

The base Qwen3-8B model will be downloaded from Hugging Face automatically.

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

3. **SimpleGameMemory** (`src/utils/streamlined_mafia_agent.py`)
   - Investigation results tracking
   - Role claims from players
   - Voting pattern analysis
   - Bounded discussion history (last 5 turns)
   - LLM observation summarization cache

4. **ModalAgent** (`src/utils/agent.py`)
   - HTTP client for Modal endpoint communication
   - Request/response handling with timeouts
   - Response parsing and formatting

5. **Modal Endpoint** (`src/utils/modal_secretmafia.py`)
   - FastAPI server running Qwen3-8B model
   - Phase-aware prompt engineering
   - Response extraction and validation
   - GPU inference with 8-bit quantization

### Decision Flow

```
Observation → Phase Detection → Game State Update → ToM Analysis → Prompt Generation → LLM Inference → Response Extraction → Action
```

## Game Logs

All games are automatically logged to `src/game_logs/` with filename format:
```
game_YYYYMMDD_HHMMSS_<role>_<outcome>.json
```

Example: `game_20251021_143000_mafia_win.json`

Each log contains:
- Full observation and action history
- Turn-by-turn analysis
- Performance metrics
- Memory state (investigation results, role claims, voting patterns)
- Error logs (if any)

## Testing

### Manual Testing

Test the agent locally:

```bash
cd src
python -c "
from utils.streamlined_mafia_agent import StreamlinedMafiaAgent
agent = StreamlinedMafiaAgent('https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run')
response = agent('Test observation')
print(response)
"
```

### Endpoint Health Check

```bash
curl https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "Enhanced SecretMafia Agent (Base Qwen3-8B)",
  "features": [
    "Strategic role-based reasoning",
    "Phase-aware response generation",
    "Enhanced prompt engineering",
    "Behavioral pattern analysis",
    "Theory of Mind reasoning"
  ]
}
```
