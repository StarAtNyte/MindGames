# Quick Start Guide - Stage 2 Submission

## Installation (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python3 run_match.py --help

# 3. Test endpoint
curl https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run/health
```

## Run a Match

```bash
# Run one game
python3 run_match.py
```

**Note:** Each run plays one game. To play multiple games, simply run the script multiple times.

## Files Overview

### Main Files
- `run_match.py` - **Entry point** (run this to play games)
- `requirements.txt` - Dependencies with pinned versions
- `SUBMISSION_README.md` - **Complete documentation** (read this first)
- `SUBMISSION_CHECKLIST.md` - Submission verification

### Source Code
- `src/online_play.py` - Game runner for online matches
- `src/utils/streamlined_mafia_agent.py` - Main agent (ToM + Strategic reasoning)
- `src/utils/theory_of_mind.py` - Advanced ToM engine
- `src/utils/bidding_system.py` - Strategic bidding system
- `src/utils/agent.py` - Base agent classes (Modal client)
- `src/utils/modal_secretmafia.py` - Model deployment (Qwen3-8B on Modal)

### Output
- `game_logs/` - Automatic game logs (created after first game)

## Key Information

**Team:** MG25-3162A7F500
**Model:** ZeroR-SecretMafia-Efficient-v4
**Base:** Qwen3-8B (8B parameters)
**Endpoint:** https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run
**Track:** Social Detection (Small Model)

## Submission

1. **GitHub Repository** (submit via form)
   - All code in repository
   - README with instructions
   - Entry point script working

2. **Google Form**
   - URL: https://docs.google.com/forms/d/1SYN2jp7qPF3V5nqWP7kgKAcNu2T5MWNE7oCZ7jFSAF0/
   - Include repository URL
   - Include Modal endpoint URL

3. **Deadline:** October 23, 2025 (23:59 ET)

## For Evaluators

### Quick Evaluation Setup

```bash
git clone <repository-url>
cd SecretMafia
pip install -r requirements.txt
python3 run_match.py  # Plays one game
```

### Health Check

```bash
curl https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run/health
```

Expected: `{"status": "healthy", ...}`

### Notes
- First request may take 30-60s (Modal cold start)
- Subsequent requests: 2-5s per turn
- Model hosted on Modal (no local GPU needed)
- Logs saved to `game_logs/`

## Troubleshooting

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**Endpoint timeout (first request):**
- Normal behavior (cold start)
- Wait 60 seconds and retry

**Import errors:**
```bash
# Make sure you're in the correct directory
cd <repository-root>
python3 run_match.py
```

## Support

See `SUBMISSION_README.md` for complete documentation.
See `SUBMISSION_CHECKLIST.md` for submission verification.

For questions: mindgameschallenge2025@gmail.com
