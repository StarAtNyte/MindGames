import textarena as ta
import time
import json
import os
from datetime import datetime
from utils.mafia_agent import EliteEnhancedSocialAgent

MODEL_NAME = "ZeroR-SecretMafia-Efficient"
MODEL_DESCRIPTION = "Elite Enhanced agent with Advanced ToM, Strategic Bidding, and Dense Rewards for Track 1 - Social Detection (SecretMafia-v0). MCTS disabled for single response generation."
team_hash = "MG25-3162A7F500" 

modal_endpoint_url = "https://khanalnitiz20--enhanced-secretmafia-fastapi-app.modal.run"
agent = EliteEnhancedSocialAgent(modal_endpoint_url=modal_endpoint_url)

# Create logs directory
LOGS_DIR = "game_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def print_banner():
    """Print beautiful startup banner"""
    print("\n" + "═" * 70)
    print("🎭" + " " * 20 + "ELITE SOCIAL DEDUCTION AGENT" + " " * 20 + "🎭")
    print("═" * 70)
    print("┌─ 🧠 Advanced Theory of Mind Engine (2nd-order beliefs)")
    print("├─ 💰 Strategic Bidding System (research-based urgency)")
    print("├─ ⚡ Single Response Generation (optimized for speed)")
    print("├─ 🎯 Dense Reward Calculation (belief-state changes)")
    print("├─ 🛡️  Enhanced Failure Mode Prevention")
    print("├─ 🧩 Reflective Memory System with behavioral analysis")
    print("├─ 🎲 Role-specific strategic heuristics (optimized)")
    print("└─ 😎 Emotion analysis and deception detection")
    print("═" * 70 + "\n")

print_banner()

env = ta.make_mgc_online(
    track="Social Detection", 
    model_name=MODEL_NAME,
    model_description=MODEL_DESCRIPTION,
    team_hash=team_hash,
    agent=agent,
    small_category=True  
)

# Play multiple games continuously
game_count = 0
max_games = 10  # Play up to 10 games, then restart script

def save_game_log(game_data):
    """Save comprehensive game log to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game_{timestamp}_{game_data.get('role', 'unknown')}_{game_data.get('outcome', 'unknown')}.json"
    filepath = os.path.join(LOGS_DIR, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2, default=str)
        print(f"✓ Game log saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Failed to save game log: {e}")
        return None

def get_performance_summary(last_n_games=5):
    """Get performance summary from recent games"""
    try:
        log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.json')]
        log_files.sort(reverse=True)  # Most recent first
        
        if not log_files:
            return "No game logs found"
        
        recent_games = []
        for filename in log_files[:last_n_games]:
            try:
                with open(os.path.join(LOGS_DIR, filename), 'r') as f:
                    game_data = json.load(f)
                    recent_games.append(game_data)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        if not recent_games:
            return "No valid game logs found"
        
        # Calculate metrics
        total_games = len(recent_games)
        wins = sum(1 for game in recent_games if game.get('outcome') == 'win')
        win_rate = wins / total_games if total_games > 0 else 0
        avg_reward = sum(game.get('reward', 0) for game in recent_games) / total_games
        
        # Role performance
        role_stats = {}
        for game in recent_games:
            role = game.get('role', 'unknown')
            if role not in role_stats:
                role_stats[role] = {'games': 0, 'wins': 0}
            role_stats[role]['games'] += 1
            if game.get('outcome') == 'win':
                role_stats[role]['wins'] += 1
        
        summary = f"""
┌─ 📊 Last {total_games} Games Summary ─────────────────────────┐
│ � Ovverall Win Rate: {win_rate:>6.1%} ({wins:>2d}/{total_games:>2d})              │
│ 💰 Average Reward:   {avg_reward:>6.2f}                      │
├─ 🎭 Role Performance: ─────────────────────────────┤"""
        
        role_emojis = {
            'Mafia': '🔪',
            'Detective': '🔍', 
            'Doctor': '⚕️',
            'Villager': '👥',
            'unknown': '❓'
        }
        
        for role, stats in role_stats.items():
            role_win_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
            emoji = role_emojis.get(role, '❓')
            summary += f"\n│ {emoji} {role:<10}: {role_win_rate:>6.1%} ({stats['wins']:>2d}/{stats['games']:>2d})              │"
        
        summary += f"\n└{'─'*50}┘"
        return summary
        
    except Exception as e:
        return f"Error generating summary: {e}"

while game_count < max_games:
    game_start_time = time.time()
    game_log = {
        "game_id": f"game_{game_count + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "game_number": game_count + 1,
        "observations": [],
        "actions": [],
        "turn_details": [],
        "agent_memory": {},
        "performance_metrics": {},
        "error_log": []
    }
    
    try:
        progress_bar = "█" * (game_count + 1) + "░" * (max_games - game_count - 1)
        print(f"\n╔{'═'*60}╗")
        print(f"║{' '*20}🎮 GAME {game_count + 1:2d} START 🎮{' '*20}║")
        print(f"╠{'═'*60}╣")
        print(f"║ Progress: [{progress_bar}] {game_count + 1:2d}/{max_games:2d} ║")
        print(f"╚{'═'*60}╝")
        print(f"🚀 Initializing enhanced social deduction gameplay...")
        env.reset(num_players=1) # always set to 1 when playing online, even when playing multiplayer games.

        done = False
        turn_count = 0
        
        while not done:
            turn_start_time = time.time()
            player_id, observation = env.get_observation()
            
            status_emoji = "🟢" if turn_count < 10 else "🟡" if turn_count < 20 else "🔴"
            print(f"\n┌─ {status_emoji} Turn {turn_count + 1:2d} ─────────────────────────────────────┐")
            obs_size = f"{len(observation):,} chars"
            print(f"│ 📝 Processing observation ({obs_size})...{' '*(25-len(obs_size))}│")
            
            # Log observation
            game_log["observations"].append({
                "turn": turn_count,
                "timestamp": datetime.now().isoformat(),
                "player_id": player_id,
                "content": observation,
                "length": len(observation)
            })
            
            # Enhanced agent processes observation with social deduction techniques
            action = agent(observation)
            
            # Calculate turn duration first
            turn_duration = time.time() - turn_start_time
            
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Show full action for short actions, preview for long ones
            if len(action) <= 100:
                print(f"│ 🎯 Decision: {action:<35}│")
            else:
                print(f"│ 🎯 Decision: {action[:100]}...│")
                print(f"│     📝 Full action ({len(action)} chars) in logs    │")
            
            print(f"│ ⏰ Time: {current_time}    Duration: {turn_duration:.1f}s{' '*15}│")
            print(f"└{'─'*50}┘")
            
            # Also print the full action separately for debugging
            if len(action) > 20:  # Only show full action if it's substantial
                print(f"\n💬 Full Action: {action}")
                print(f"{'─'*60}")
            
            # Log action and turn details
            game_log["actions"].append({
                "turn": turn_count,
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "processing_time": turn_duration
            })
            
            game_log["turn_details"].append({
                "turn": turn_count,
                "player_id": player_id,
                "observation_length": len(observation),
                "action_length": len(action),
                "processing_time": turn_duration,
                "memory_entries": len(agent.memory.observational) if hasattr(agent, 'memory') else 0,
                "discussion_entries": len(agent.memory.discussion_history) if hasattr(agent, 'memory') else 0
            })
            
            done, step_info = env.step(action=action)
            turn_count += 1

        rewards, game_info = env.close()
        game_count += 1
        
        # Calculate game duration and outcome
        game_duration = time.time() - game_start_time
        
        # Handle rewards properly - it's a dictionary, extract the actual reward value
        if isinstance(rewards, dict):
            # Get the reward for the current player or sum all rewards
            reward_value = sum(rewards.values()) if rewards else 0
        else:
            reward_value = rewards if rewards is not None else 0
            
        outcome = "win" if reward_value > 0 else "loss"
        
        print(f"\n╔{'═'*60}╗")
        print(f"║{' '*18}🏁 GAME {game_count} COMPLETE 🏁{' '*18}║")
        print(f"╠{'═'*60}╣")
        print(f"║ 💰 Final Reward: {reward_value:>8.2f}{' '*32}║")
        print(f"║ 🔄 Total Turns:  {turn_count:>8d}{' '*32}║")
        print(f"║ ⏱️  Duration:     {game_duration:>8.1f}s{' '*31}║")
        avg_turn_time = game_duration/turn_count if turn_count > 0 else 0
        print(f"║ ⚡ Avg Turn Time: {avg_turn_time:>8.1f}s{' '*31}║")
        print(f"╚{'═'*60}╝")
        
        # Update final performance metrics
        if outcome == 'win':
            if hasattr(agent, 'my_role') and agent.my_role and agent.my_role.value == 'Mafia':
                game_outcome_str = "Mafia wins"
            else:
                game_outcome_str = "Village wins"
        else:
            game_outcome_str = "Game ended"
        
        agent.update_final_metrics(game_outcome_str, reward_value)
        
        # Generate end-of-game summary
        agent.generate_round_summary()
        
        # ============================================================================
        # CHAMPIONSHIP ENHANCEMENTS: Performance Summary
        # ============================================================================
        
        # Print championship enhancement status
        if hasattr(agent, 'print_championship_status'):
            agent.print_championship_status()
        
        # Extract agent memory and role information
        agent_memory = {}
        agent_role = "unknown"
        agent_player_id = -1
        
        if hasattr(agent, 'memory'):
            agent_memory = {
                "observational": agent.memory.observational,
                "reflective": agent.memory.reflective,
                "voting_patterns": agent.memory.voting_patterns,
                "suspicion_tracker": agent.memory.suspicion_tracker,
                "discussion_history": agent.memory.discussion_history
            }
        
        if hasattr(agent, 'my_role'):
            agent_role = agent.my_role
        if hasattr(agent, 'my_player_id'):
            agent_player_id = agent.my_player_id
        
        # Complete game log
        game_log.update({
            "outcome": outcome,
            "reward": reward_value,
            "raw_rewards": rewards,  # Keep the original rewards dict for debugging
            "total_turns": turn_count,
            "game_duration": game_duration,
            "role": agent_role,
            "player_id": agent_player_id,
            "agent_memory": agent_memory,
            "game_info": game_info if game_info else {},
            "performance_metrics": {
                "turns_per_minute": turn_count / (game_duration / 60) if game_duration > 0 else 0,
                "avg_turn_time": game_duration / turn_count if turn_count > 0 else 0,
                "memory_growth": len(agent.memory.observational) if hasattr(agent, 'memory') else 0,
                "discussion_participation": len(agent.memory.discussion_history) if hasattr(agent, 'memory') else 0,
                "reward_per_turn": reward_value / turn_count if turn_count > 0 else 0
            }
        })
        
        # Save game log
        save_game_log(game_log)
        
        # Memory and performance stats
        memory_entries = len(agent.memory.observational) if hasattr(agent, 'memory') else 0
        discussion_entries = len(agent.memory.discussion_history) if hasattr(agent, 'memory') else 0
        investigation_entries = len(agent.memory.investigation_results) if hasattr(agent, 'memory') else 0
        role_claims = len(agent.memory.role_claims) if hasattr(agent, 'memory') else 0
        total_rewards = agent.performance_metrics.get('total_rewards', 0)
        
        print(f"\n┌─ 📊 PERFORMANCE SUMMARY ─────────────────────────┐")
        print(f"│ � Memory  Entries:       {memory_entries:>6d}                │")
        print(f"│ 💬 Discussion History:   {discussion_entries:>6d}                │")
        print(f"│ 🔍 Investigation Results: {investigation_entries:>6d}                │")
        print(f"│ 🎭 Role Claims Tracked:  {role_claims:>6d}                │")
        print(f"│ 🎯 Total Rewards:        {total_rewards:>6.2f}                │")
        print(f"└{'─'*50}┘")
        
        # Show performance summary every 3 games
        if game_count % 3 == 0:
            print(f"\n╔{'═'*60}╗")
            print(f"║{' '*18}📈 PERFORMANCE REPORT 📈{' '*18}║")
            print(f"╚{'═'*60}╝")
            print(get_performance_summary())
            
            # Show championship enhancement summary
            if hasattr(agent, 'get_championship_performance_summary'):
                championship_summary = agent.get_championship_performance_summary()
                print(f"\n┌─ 🏆 CHAMPIONSHIP ENHANCEMENTS ───────────────────┐")
                print(f"│ 🛡️  Security:      {championship_summary['security_metrics']['jailbreak_attempts_blocked']:>6d} blocks           │")
                print(f"│ 🤝 Communication:  {championship_summary['communication_metrics']['diplomatic_enhancements_applied']:>6d} enhancements    │")
                print(f"│ 🎯 Strategy:       {championship_summary['strategic_metrics']['influence_success_rate']:>6.1%} influence rate  │")
                print(f"│ ⭐ Overall Score:  {championship_summary['overall_enhancement_score']:>6.2f}/2.00             │")
                print(f"└{'─'*50}┘")
        
    except Exception as e:
        print(f"\n╔{'═'*60}╗")
        print(f"║{' '*22}💥 GAME ERROR 💥{' '*22}║")
        print(f"╠{'═'*60}╣")
        print(f"║ ❌ Game {game_count + 1} Error: {str(e)[:40]:<40}║")
        import traceback
        error_traceback = traceback.format_exc()
        print(f"║ 🔍 Stack Trace (see logs for full details){' '*12}║")
        print(f"╚{'═'*60}╝")
        print(f"\n🔍 Full Error Details:")
        print(f"{'─'*60}")
        print(error_traceback)
        print(f"{'─'*60}")
        
        # Log error
        game_log["error_log"].append({
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": error_traceback,
            "turn": turn_count
        })
        
        # Save partial game log even on error
        game_log.update({
            "outcome": "error",
            "reward": 0,
            "raw_rewards": {},
            "total_turns": turn_count,
            "game_duration": time.time() - game_start_time,
            "role": getattr(agent, 'my_role', 'unknown'),
            "player_id": getattr(agent, 'my_player_id', -1)
        })
        save_game_log(game_log)
        
        # If it's a server connection error, wait and continue
        if "server shutdown" in str(e) or "No valid observation" in str(e):
            print(f"\n┌─ 🔄 CONNECTION RECOVERY ─────────────────────────┐")
            print(f"│ 📡 Server connection lost, retrying in 5s...    │")
            print(f"└{'─'*50}┘")
            time.sleep(5)
            continue
        else:
            break

print(f"\n╔{'═'*60}╗")
print(f"║{' '*19}🎊 SESSION COMPLETE 🎊{' '*19}║")
print(f"╠{'═'*60}╣")
print(f"║ ✅ Completed {game_count:>2d} Elite Enhanced Games{' '*22}║")
print(f"║ 🔄 Restarting script to continue playing...{' '*12}║")
print(f"╚{'═'*60}╝")

print(f"\n╔{'═'*60}╗")
print(f"║{' '*18}🏆 FINAL PERFORMANCE 🏆{' '*18}║")
print(f"╚{'═'*60}╝")
print(get_performance_summary(10))  # Show final summary of last 10 games
print(f"\n{'🎊'*20}")
print(f"{'═'*60}")
print(f"{'🎊'*20}")


