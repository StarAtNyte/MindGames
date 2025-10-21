import textarena as ta
import time
import json
import os
from datetime import datetime
from utils.streamlined_mafia_agent import StreamlinedMafiaAgent

MODEL_NAME = "ZeroR-SecretMafia-Efficient-v4"
MODEL_DESCRIPTION = "Qwen3-8B with Theory of Mind reasoning and two-step decision making. Pure LLM agent with no heuristics."
team_hash = "MG25-3162A7F500" 

modal_endpoint_url = "https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run"
agent = StreamlinedMafiaAgent(modal_endpoint_url=modal_endpoint_url)

# Create logs directory
LOGS_DIR = "game_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 70)
    print("SecretMafia Agent - Stage 2 Submission")
    print("Model: " + MODEL_NAME)
    print("Team: " + team_hash)
    print("=" * 70 + "\n")

print_banner()

env = ta.make_mgc_online(
    track="Social Detection", 
    model_name=MODEL_NAME,
    model_description=MODEL_DESCRIPTION,
    team_hash=team_hash,
    agent=agent,
    small_category=True  
)

# Play one game per run
game_count = 0
max_games = 1  # Play one game per run

def save_game_log(game_data):
    """Save comprehensive game log to JSON file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Safely get role and outcome with fallbacks
        role = game_data.get('role', 'unknown') if game_data else 'unknown'
        outcome = game_data.get('outcome', 'unknown') if game_data else 'unknown'
        
        # Clean filename - replace any problematic characters
        role_clean = str(role).replace(' ', '_').replace('/', '_') if role else 'unknown'
        outcome_clean = str(outcome).replace(' ', '_').replace('/', '_') if outcome else 'unknown'
        
        filename = f"game_{timestamp}_{role_clean}_{outcome_clean}.json"
        filepath = os.path.join(LOGS_DIR, filename)
        
        # Ensure game_data is not None
        if game_data is None:
            game_data = {"error": "No game data available", "timestamp": timestamp}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, default=str, ensure_ascii=False)
        print(f"✓ Game log saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Failed to save game log: {e}")
        # Try to save a minimal error log
        try:
            error_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_filename = f"game_{error_timestamp}_error_error.json"
            error_filepath = os.path.join(LOGS_DIR, error_filename)
            error_data = {"error": str(e), "timestamp": error_timestamp, "original_data": str(game_data)}
            with open(error_filepath, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, default=str)
            print(f"✓ Error log saved: {error_filepath}")
        except:
            pass
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
        
        # Calculate metrics with safe defaults
        total_games = len(recent_games)
        wins = sum(1 for game in recent_games if game.get('outcome') == 'win')
        win_rate = wins / total_games if total_games > 0 else 0.0
        
        # Calculate average reward safely
        total_reward = sum(game.get('reward', 0) for game in recent_games if game.get('reward') is not None)
        avg_reward = total_reward / total_games if total_games > 0 else 0.0
        
        # Role performance
        role_stats = {}
        for game in recent_games:
            role = game.get('role', 'unknown')
            if role not in role_stats:
                role_stats[role] = {'games': 0, 'wins': 0}
            role_stats[role]['games'] += 1
            if game.get('outcome') == 'win':
                role_stats[role]['wins'] += 1
        
        # Build summary with safe formatting
        try:
            summary = f"""
┌─ 📊 Last {total_games} Games Summary ─────────────────────────┐
│ 🎯 Overall Win Rate: {win_rate:>6.1%} ({wins:>2d}/{total_games:>2d})              │
│ 💰 Average Reward:   {avg_reward:>6.2f}                      │
├─ 🎭 Role Performance: ─────────────────────────────┤"""
        except Exception as format_error:
            return f"Summary formatting error: {format_error}"
        
        role_emojis = {
            'Mafia': '🔪',
            'Detective': '🔍', 
            'Doctor': '⚕️',
            'Villager': '👥',
            'unknown': '❓'
        }
        
        # Add role performance with error handling
        try:
            for role, stats in role_stats.items():
                if stats and isinstance(stats, dict) and stats.get('games', 0) > 0:
                    wins = stats.get('wins', 0) 
                    games = stats.get('games', 0)
                    role_win_rate = wins / games if games > 0 else 0.0
                    emoji = role_emojis.get(role, '❓')
                    role_name = str(role) if role is not None else 'unknown'
                    summary += f"\n│ {emoji} {role_name:<10}: {role_win_rate:>6.1%} ({wins:>2d}/{games:>2d})              │"
            
            summary += f"\n└{'─'*50}┘"
            return summary
        except Exception as role_error:
            return f"Role stats formatting error: {role_error}"
        
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
                "memory_entries": len(agent.memory.investigation_results) if hasattr(agent, 'memory') else 0,
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
            if hasattr(agent, 'my_role') and agent.my_role == 'mafia':
                game_outcome_str = "Mafia wins"
            else:
                game_outcome_str = "Village wins"
        else:
            game_outcome_str = "Game ended"
        
        # Update agent's total rewards
        if hasattr(agent, 'total_rewards'):
            agent.total_rewards += reward_value
        
        # ============================================================================
        # CHAMPIONSHIP ENHANCEMENTS: Performance Summary
        # ============================================================================
        
        # Streamlined agent doesn't have championship status
        
        # Extract agent memory and role information
        agent_memory = {}
        agent_role = "unknown"
        agent_player_id = -1
        
        if hasattr(agent, 'memory'):
            agent_memory = {
                "investigation_results": agent.memory.investigation_results,
                "role_claims": agent.memory.role_claims,
                "voting_patterns": agent.memory.voting_patterns,
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
                "memory_growth": len(agent.memory.investigation_results) if hasattr(agent, 'memory') else 0,
                "discussion_participation": len(agent.memory.discussion_history) if hasattr(agent, 'memory') else 0,
                "reward_per_turn": reward_value / turn_count if turn_count > 0 else 0
            }
        })
        
        # Save game log
        save_game_log(game_log)
        
        # Memory and performance stats
        memory_entries = len(agent.memory.investigation_results) if hasattr(agent, 'memory') else 0
        discussion_entries = len(agent.memory.discussion_history) if hasattr(agent, 'memory') else 0
        investigation_entries = len(agent.memory.investigation_results) if hasattr(agent, 'memory') else 0
        role_claims = len(agent.memory.role_claims) if hasattr(agent, 'memory') else 0
        total_rewards = getattr(agent, 'total_rewards', 0)
        
        print(f"\n┌─ 📊 PERFORMANCE SUMMARY ─────────────────────────┐")
        print(f"│ 📚 Memory Entries:       {memory_entries:>6d}                │")
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
            
            # Show streamlined agent performance
            strategic_decisions = getattr(agent, 'strategic_decisions', 0)
            print(f"\n┌─ 🎯 STREAMLINED PERFORMANCE ─────────────────────┐")
            print(f"│ 🧠 ToM Analyses:     {agent.turn_count:>6d} turns           │")
            print(f"│ 💰 Strategic Decisions: {strategic_decisions:>6d} made            │")
            print(f"│ 🎯 Total Rewards:    {getattr(agent, 'total_rewards', 0):>6.2f} earned          │")
            print(f"│ ⚡ Rules Compliant:   100% LLM reasoning     │")
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
print(f"║ ✅ Completed {game_count:>2d} Streamlined Games{' '*25}║")
print(f"║ 🔄 Restarting script to continue playing...{' '*12}║")
print(f"╚{'═'*60}╝")

print(f"\n╔{'═'*60}╗")
print(f"║{' '*18}🏆 FINAL PERFORMANCE 🏆{' '*18}║")
print(f"╚{'═'*60}╝")
print(get_performance_summary(10))  # Show final summary of last 10 games
print(f"\n{'🎊'*20}")
print(f"{'═'*60}")
print(f"{'🎊'*20}")

