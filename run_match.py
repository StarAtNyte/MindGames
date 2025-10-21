import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_online_match():
    """Run one online match via TextArena MGC platform"""
    import textarena as ta
    import time
    import json
    from datetime import datetime
    from utils.streamlined_mafia_agent import StreamlinedMafiaAgent

    MODEL_NAME = "ZeroR-SecretMafia-Efficient-v4"
    MODEL_DESCRIPTION = "Streamlined agent with Theory of Mind, Strategic Bidding, Rules Compliant - no heuristics for Track 1 - Social Detection (SecretMafia-v0)."
    team_hash = "MG25-3162A7F500"
    modal_endpoint_url = "https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run"

    # Create agent
    agent = StreamlinedMafiaAgent(modal_endpoint_url=modal_endpoint_url)

    # Create logs directory
    LOGS_DIR = "game_logs"
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("SecretMafia Agent - Stage 2 Submission")
    print("Model: ZeroR-SecretMafia-Efficient-v4")
    print("Team: MG25-3162A7F500")
    print("=" * 70 + "\n")

    # Create environment (plays one game)
    env = ta.make_mgc_online(
        track="Social Detection",
        model_name=MODEL_NAME,
        model_description=MODEL_DESCRIPTION,
        team_hash=team_hash,
        agent=agent,
        small_category=True
    )

    print("\nStarting game...")

    try:
        env.reset(num_players=1)
        done = False
        turn_count = 0

        while not done:
            turn_count += 1
            player_id, observation = env.get_observation()
            print(f"\nTurn {turn_count}: Processing observation...")

            action = agent(observation)
            action_preview = action[:50] + "..." if len(action) > 50 else action
            print(f"Action: {action_preview}")

            done, step_info = env.step(action=action)

        rewards, game_info = env.close()

        # Extract reward value
        if isinstance(rewards, dict):
            reward_value = sum(rewards.values()) if rewards else 0
        else:
            reward_value = rewards if rewards is not None else 0

        outcome = "win" if reward_value > 0 else "loss"

        print("\n" + "=" * 60)
        print("Game Complete")
        print("=" * 60)
        print(f"Reward:  {reward_value:.2f}")
        print(f"Turns:   {turn_count}")
        print(f"Outcome: {outcome}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\nSession complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Run one SecretMafia game with ZeroR agent"
    )

    args = parser.parse_args()
    run_online_match()


if __name__ == "__main__":
    main()
