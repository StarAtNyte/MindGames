# Secret Mafia AI Agent

A Mafia game AI agent that plays social deduction games. Built for the MindGames competition.

## What it does

This agent plays Secret Mafia games online. It can take any role (Villager, Doctor, Detective, or Mafia) and tries to win by analyzing other players' behavior and making strategic decisions.

## Key features

- **Theory of Mind**: Tracks what players think about each other
- **Strategic reasoning**: Uses game theory to make decisions  
- **Role adaptation**: Changes strategy based on assigned role
- **Behavioral analysis**: Looks for suspicious patterns in player actions
- **Memory system**: Remembers important game events and player behavior

## How to run

1. Install dependencies:
```bash
pip install textarena
```

2. Set up your Modal endpoint URL in the code

3. Run the agent:
```bash
cd src
python online_play_streamlined.py
```

## Files

- `src/online_play_streamlined.py` - Main game runner
- `src/utils/streamlined_mafia_agent.py` - Core agent logic
- `src/utils/theory_of_mind.py` - Theory of Mind engine
- `src/utils/bidding_system.py` - Strategic bidding system
- `game_logs/` - Automatic game logs

## How it works

The agent analyzes player statements and voting patterns to identify Mafia members. It uses different strategies depending on its role:

- **Villager**: Looks for suspicious behavior and votes to eliminate Mafia
- **Doctor**: Protects other players from elimination
- **Detective**: Investigates players to confirm their roles
- **Mafia**: Blends in while eliminating Village team members

Each game is automatically logged with detailed performance data.

## Requirements

- Python 3.8+
- TextArena environment
- Modal endpoint for LLM integration