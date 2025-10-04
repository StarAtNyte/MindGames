# Elite Enhanced Social Deduction Agent

A state-of-the-art AI agent for social deduction games, specifically designed for Secret Mafia competitions. This agent combines cutting-edge research in game theory, psychology, and AI to achieve exceptional performance across all game roles.

## 🏆 Performance Highlights

- **100% Win Rate** across 3 games (Villager, Mafia, Doctor)
- **TrueSkill Rating**: 42.605 (started at 25.0)
- **Multi-Role Mastery**: Successfully adapts strategy based on assigned role
- **Advanced AI Components**: Theory of Mind, MCTS Planning, Strategic Bidding

## 🧠 Core Technologies

### 1. Advanced Theory of Mind Engine
- **Second-order belief modeling**: Tracks what players think about other players
- **Behavioral pattern analysis**: Identifies Mafia tells through language patterns
- **Dynamic suspicion tracking**: Real-time updates based on player actions
- **Meta-belief systems**: Models complex social dynamics

### 2. MCTS Communication Planning
- **500-iteration Monte Carlo Tree Search** for optimal response selection
- **Multi-factor reward calculation** based on belief state changes
- **Strategic response optimization** using game theory principles
- **Research-backed decision making** from MultiMind studies

### 3. Strategic Bidding System
- **Research-based urgency calculation** for optimal turn-taking
- **Context-aware bidding** responds to mentions and accusations
- **Dynamic priority adjustment** based on game state
- **40% mentioned player urgency** (based on Werewolf Arena findings)

### 4. Dense Reward Calculator
- **Belief-state change tracking** for continuous learning
- **Multi-dimensional performance metrics**:
  - Suspicion reduction on self
  - Mafia suspicion increase
  - Information sharing value
  - Strategic positioning rewards

### 5. Enhanced Failure Prevention
- **Natural language variation** prevents repetitive responses
- **Role leakage prevention** protects secret information
- **Response history tracking** avoids suspicious patterns
- **Adaptive communication style** based on game context

## 🎮 Game Features

### Role-Specific Strategies

#### 🏘️ **Villager**
- Aggressive analytical approach (research-proven to win more)
- Pattern recognition for Mafia identification
- Coordinated voting with other Village roles
- Information gathering and sharing

#### 🕵️ **Detective**
- Strategic investigation targeting using ToM engine
- Optimal timing for role reveals
- Evidence-based accusation strategies
- Protection coordination with Doctor

#### 🏥 **Doctor**
- Predictive protection using behavioral analysis
- Key player identification and safeguarding
- Strategic healing decisions
- Threat assessment algorithms

#### 🔫 **Mafia**
- Sophisticated deception and misdirection
- Teammate coordination without obvious tells
- Village role hunting and elimination
- Blend-in tactics with helpful-appearing responses

## 📊 Performance Metrics

The agent tracks comprehensive performance data:

```python
performance_metrics = {
    'total_rewards': 4.08,           # Cumulative game rewards
    'strategic_decisions': 2,        # Quality decision count
    'information_contributions': 1,   # Analytical insights shared
    'successful_deflections': 0,     # Defense against accusations
    'belief_accuracy': 0.1          # Prediction accuracy rate
}
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TextArena environment
- Modal endpoint (for LLM integration)

### Installation
```bash
git clone <repository>
cd MindGames
pip install -r requirements.txt
```

### Running the Agent
```bash
cd src
python online_play_track1.py
```

### Configuration
Update the Modal endpoint URL in `online_play_track1.py`:
```python
modal_endpoint_url = "your-modal-endpoint-here"
```

## 📁 Project Structure

```
MindGames/
├── src/
│   ├── online_play_track1.py          # Main game runner
│   └── utils/
│       ├── mafia_agent.py             # Core agent implementation
│       ├── theory_of_mind.py          # ToM engine
│       ├── bidding_system.py          # Strategic bidding
│       ├── communication_planner.py   # MCTS planner
│       └── agent.py                   # Base agent class
├── game_logs/                         # Automatic game logging
└── README.md
```

## 🔧 Advanced Features

### Automatic Game Logging
Every game is automatically logged with comprehensive data:
- Turn-by-turn observations and actions
- Performance metrics tracking
- Strategic decision analysis
- Memory system snapshots

### Real-time Performance Analysis
```python
# Performance summary every 3 games
def get_performance_summary(last_n_games=5):
    # Calculates win rates, role performance, trends
```

### Enhanced Memory System
- **Observational memory**: Raw game observations
- **Reflective memory**: Strategic insights and summaries
- **Voting patterns**: Historical decision tracking
- **Behavioral analysis**: Player pattern recognition

## 🧪 Research Foundation

This agent is built on cutting-edge research:

- **MultiMind Framework**: Advanced Theory of Mind modeling
- **Werewolf Arena Studies**: Optimal bidding strategies
- **Social Deduction Psychology**: Behavioral pattern recognition
- **Game Theory Optimization**: MCTS for strategic planning

## 📈 Performance Optimization

### Continuous Improvement
- Belief accuracy tracking with real-time updates
- Strategic decision quality measurement
- Role-specific performance analysis
- Adaptive learning from game outcomes

### Debugging and Monitoring
- Comprehensive phase detection logging
- Real-time performance metric updates
- Strategic decision reasoning output
- Memory system state tracking

## 🎯 Future Enhancements

- [ ] Multi-game learning and adaptation
- [ ] Advanced alliance detection algorithms
- [ ] Emotional state modeling for deception detection
- [ ] Cross-game strategy transfer learning
- [ ] Tournament-specific optimizations

## 📝 License

This project is part of the MindGames competition framework.

## 🤝 Contributing

This is a competition entry, but insights and improvements are welcome for educational purposes.

---

**Elite Enhanced Social Deduction Agent** - Where AI meets psychology in the ultimate game of deception and deduction.