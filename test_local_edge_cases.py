import sys
import os
import time
import re

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.local_llm_agent import LocalStreamlinedMafiaAgent

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
USE_4BIT = True


class LocalEdgeCaseTester:
    """Test all possible edge cases and failure modes for LOCAL LLM agent"""

    def __init__(self, model_name: str, use_4bit: bool = True):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.test_results = {}
        self.agent = None

    def initialize_agent(self):
        """Initialize the local agent once"""
        if self.agent is None:
            print("\n🚀 Initializing Local LLM Agent...")
            print(f"   Model: {self.model_name}")
            print(f"   Quantization: {'4-bit' if self.use_4bit else 'Full precision'}")
            self.agent = LocalStreamlinedMafiaAgent(
                model_name=self.model_name,
                load_in_4bit=self.use_4bit
            )
            print("   ✅ Agent initialized successfully")

    def test_invalid_observations(self) -> bool:
        """Test handling of corrupted/invalid observations"""
        print("\n🔴 Testing Invalid Observation Handling...")
        print("-" * 60)

        test_cases = [
            ("Empty string", ""),
            ("Whitespace only", "   \n\n   "),
            ("Garbage data", "###CORRUPTED_DATA_!@#$%^&*()"),
            ("Very long observation", "x" * 50000),
            ("Unicode chaos", "🎭💀🔪👥" * 100),
            ("Null bytes", "Player\x001 is\x00suspicious"),
        ]

        all_passed = True

        for name, obs in test_cases:
            print(f"\n   Testing: {name}")
            try:
                self.agent.my_role = "villager"
                self.agent.my_player_id = 1
                self.agent.alive_players = [0, 1, 2, 3]
                self.agent.turn_count = 2

                # Should not crash, should return something
                response = self.agent(obs)

                if response and len(response) > 0:
                    print(f"      ✅ Handled gracefully: '{response[:50]}...'")
                else:
                    print(f"      ❌ Returned empty response")
                    all_passed = False

            except Exception as e:
                print(f"      ⚠️  Exception (may be acceptable): {e}")
                # Not necessarily a failure - some garbage input should error

        return all_passed

    def test_timeout_scenarios(self) -> bool:
        """Test behavior under timeout pressure"""
        print("\n⏱️  Testing Timeout Scenarios...")
        print("-" * 60)

        try:
            self.agent.my_role = "mafia"
            self.agent.my_player_id = 0
            self.agent.alive_players = [0, 1, 2, 3, 4]
            self.agent.turn_count = 2

            # Very complex scenario that might take long
            obs = """Day 15, Turn 47. This is a critical moment.
Player 1: After analyzing the voting patterns from turns 1-46, cross-referencing the night kill sequences,
examining the discussion dynamics, reviewing all role claims, and considering the meta-game implications,
I believe we need to carefully evaluate each remaining player's behavioral consistency across all previous
interactions. Let me outline my 20-point analysis: First, Player 0 voted in a suspicious pattern on Day 3...
Player 2: I completely disagree with that entire analysis and here's my counter-argument in 25 points...
Player 3: Both of you are wrong, let me explain my comprehensive theory spanning all 15 days...
Player 4: Wait, I have evidence that contradicts everything said so far..."""

            print("\n   Generating response (may timeout with two-step reasoning)...")
            start = time.time()

            try:
                response = self.agent(obs)
                elapsed = time.time() - start

                print(f"   ✅ Completed in {elapsed:.1f}s")
                print(f"   Response: '{response[:80]}...'")

                if elapsed > 120:
                    print(f"   ⚠️  Slow response (>{120}s) - might timeout in real games")
                    return False

                return True

            except Exception as e:
                elapsed = time.time() - start
                print(f"   ❌ Timeout/Error after {elapsed:.1f}s: {e}")
                return False

        except Exception as e:
            print(f"❌ Timeout test setup failed: {e}")
            return False

    def test_consecutive_games_stability(self) -> bool:
        """Test running multiple games consecutively without crashes"""
        print("\n🔄 Testing Consecutive Games Stability...")
        print("-" * 60)

        num_games = 5
        failures = 0

        for game_num in range(num_games):
            print(f"\n   Game {game_num + 1}/{num_games}:")

            try:
                # Reset agent state for new game
                self.agent.my_role = ["mafia", "detective", "doctor", "villager"][game_num % 4]
                self.agent.my_player_id = game_num % 4
                self.agent.alive_players = [0, 1, 2, 3]
                self.agent.turn_count = game_num

                obs = f"""Day breaks. Game {game_num + 1}.
Player 0: Let's find the Mafia.
Player 1: I agree.
Player 2: Who should we vote for?"""

                response = self.agent(obs)

                if len(response) > 10:
                    print(f"      ✅ Game {game_num + 1} successful")
                else:
                    print(f"      ❌ Game {game_num + 1} failed - bad response")
                    failures += 1

            except Exception as e:
                print(f"      ❌ Game {game_num + 1} crashed: {e}")
                failures += 1

        success_rate = (num_games - failures) / num_games
        print(f"\n   Success rate: {num_games - failures}/{num_games} ({success_rate*100:.0f}%)")

        return failures == 0

    def test_long_game_stability(self) -> bool:
        """Test agent stability over 20+ turns"""
        print("\n📏 Testing Long Game Stability (20 turns)...")
        print("-" * 60)

        try:
            self.agent.my_role = "detective"
            self.agent.my_player_id = 1
            self.agent.alive_players = [0, 1, 2, 3]

            failures = 0

            for turn in range(1, 21):
                self.agent.turn_count = turn

                obs = f"""Turn {turn}/20. Day breaks.
Player 0: This is turn {turn}.
Player 2: We're still looking for Mafia.
Player 3: Let's vote carefully."""

                try:
                    response = self.agent(obs)

                    if len(response) > 10:
                        if turn % 5 == 0:
                            print(f"   ✅ Turn {turn}: OK")
                    else:
                        print(f"   ❌ Turn {turn}: Bad response")
                        failures += 1

                except Exception as e:
                    print(f"   ❌ Turn {turn}: Error - {e}")
                    failures += 1

            success_rate = (20 - failures) / 20
            print(f"\n   Success rate: {20 - failures}/20 ({success_rate*100:.0f}%)")

            return failures <= 2  # Allow 2 failures out of 20

        except Exception as e:
            print(f"❌ Long game test failed: {e}")
            return False

    def test_response_format_validation(self) -> bool:
        """Ensure responses ALWAYS match expected format"""
        print("\n✅ Testing Response Format Validation...")
        print("-" * 60)

        test_cases = [
            {
                "name": "Discussion Phase",
                "obs": "Day breaks. Discuss for 3 rounds.\nPlayer 0: Who is suspicious?",
                "expected_format": "discussion",
                "should_not_match": r'^\s*\[\d+\]\s*$'
            },
            {
                "name": "Voting Phase",
                "obs": "Voting phase - submit one vote in format [X]. Valid: [0], [2], [3]",
                "expected_format": "action",
                "should_match": r'^\s*\[\d+\]\s*$'
            },
            {
                "name": "Night Phase (Detective)",
                "obs": "Night phase - choose one player to investigate: [0], [2], [3]",
                "expected_format": "action",
                "should_match": r'^\s*\[\d+\]\s*$'
            },
            {
                "name": "Night Phase (Doctor)",
                "obs": "Night phase - choose one player to protect: [0], [1], [3]",
                "expected_format": "action",
                "should_match": r'^\s*\[\d+\]\s*$'
            },
            {
                "name": "Night Phase (Mafia)",
                "obs": "Night has fallen. Mafia, agree on a victim. Valid targets: [1], [2], [3]",
                "expected_format": "action",
                "should_match": r'^\s*\[\d+\]\s*$'
            },
        ]

        all_passed = True

        for i, case in enumerate(test_cases):
            print(f"\n   {i+1}. {case['name']}:")

            try:
                self.agent.my_role = "detective" if "Detective" in case['name'] else "villager"
                self.agent.my_player_id = 1
                self.agent.alive_players = [0, 1, 2, 3]
                self.agent.turn_count = 2

                response = self.agent(case['obs'])

                if case['expected_format'] == 'action':
                    if re.match(case.get('should_match', r'^\s*\[\d+\]\s*$'), response.strip()):
                        print(f"      ✅ Correct format: {response}")
                    else:
                        print(f"      ❌ Wrong format - expected [X], got: '{response}'")
                        all_passed = False

                elif case['expected_format'] == 'discussion':
                    if not re.match(case.get('should_not_match', r'^\s*\[\d+\]\s*$'), response.strip()):
                        print(f"      ✅ Correct format (discussion): '{response[:50]}...'")
                    else:
                        print(f"      ❌ Wrong format - expected discussion, got: '{response}'")
                        all_passed = False

            except Exception as e:
                print(f"      ❌ Error: {e}")
                all_passed = False

        return all_passed

    def test_player_count_variations(self) -> bool:
        """Test with different player counts"""
        print("\n👥 Testing Different Player Counts...")
        print("-" * 60)

        player_counts = [3, 4, 5, 6, 7, 8, 10, 15]
        all_passed = True

        for count in player_counts:
            print(f"\n   Testing with {count} players:")

            try:
                self.agent.my_role = "villager"
                self.agent.my_player_id = 0
                self.agent.alive_players = list(range(count))
                self.agent.turn_count = 1

                obs = f"Day breaks. {count} players alive. Discuss."

                response = self.agent(obs)

                if len(response) > 10:
                    print(f"      ✅ Handled {count} players successfully")
                else:
                    print(f"      ❌ Failed with {count} players")
                    all_passed = False

            except Exception as e:
                print(f"      ❌ Error with {count} players: {e}")
                all_passed = False

        return all_passed

    def test_role_reveal_safety(self) -> bool:
        """Test that roles don't accidentally reveal themselves"""
        print("\n🔒 Testing Role Protection (No Accidental Reveals)...")
        print("-" * 60)

        sensitive_roles = [
            ("detective", ["detective", "investigated", "checked"]),
            ("doctor", ["doctor", "protected", "saved"]),
        ]

        all_passed = True

        for role, forbidden_words in sensitive_roles:
            print(f"\n   Testing {role.upper()} protection:")

            try:
                self.agent.my_role = role
                self.agent.my_player_id = 1
                self.agent.alive_players = [0, 1, 2, 3, 4]
                self.agent.turn_count = 2

                obs = """Day breaks. Discuss for 3 rounds.
Player 0: Who do you think is suspicious?
Player 2: I'm not sure yet.
Player 3: We need more information."""

                response = self.agent(obs).lower()

                # Check for role reveals
                revealed = False
                for word in forbidden_words:
                    if word in response and "i'm" in response[:50].lower():
                        print(f"      ⚠️  Possible role reveal detected: '{word}' in response")
                        print(f"      Response: '{response[:100]}...'")
                        revealed = True

                if not revealed:
                    print(f"      ✅ No obvious role reveal")
                else:
                    all_passed = False

            except Exception as e:
                print(f"      ❌ Error: {e}")
                all_passed = False

        return all_passed

    def test_invalid_move_recovery(self) -> bool:
        """Test recovery from invalid move penalties"""
        print("\n🚨 Testing Invalid Move Recovery...")
        print("-" * 60)

        try:
            self.agent.my_role = "villager"
            self.agent.my_player_id = 1
            self.agent.alive_players = [0, 1, 2, 3]
            self.agent.turn_count = 5

            obs = """Player 1 attempted an invalid move. Reason: Vote not in valid format or invalid target.
Please resubmit a valid move and remember to follow the game rules to avoid penalties.

Voting phase - submit one vote in format [X]. Valid: [0], [2], [3]"""

            response = self.agent(obs)

            # Should recover and output valid format
            if re.match(r'^\s*\[\d+\]\s*$', response.strip()):
                target = int(re.match(r'^\s*\[(\d+)\]\s*$', response.strip()).group(1))
                if target in [0, 2, 3]:
                    print(f"   ✅ Recovered and output valid vote: {response}")
                    return True
                else:
                    print(f"   ❌ Invalid target: {target}")
                    return False
            else:
                print(f"   ❌ Still invalid format: '{response}'")
                return False

        except Exception as e:
            print(f"❌ Recovery test failed: {e}")
            return False

    def test_memory_overflow_protection(self) -> bool:
        """Test that memory doesn't overflow in long games"""
        print("\n💾 Testing Memory Overflow Protection...")
        print("-" * 60)

        try:
            self.agent.my_role = "detective"
            self.agent.my_player_id = 1
            self.agent.alive_players = [0, 1, 2, 3]

            # Simulate 30 turns with investigations
            for turn in range(30):
                self.agent.turn_count = turn
                self.agent.memory.investigation_results[turn % 4] = "MAFIA" if turn % 2 == 0 else "INNOCENT"
                self.agent.memory.discussion_history.append(f"Turn {turn} discussion" * 10)

            # Check memory sizes
            investigation_count = len(self.agent.memory.investigation_results)
            discussion_count = len(self.agent.memory.discussion_history)

            print(f"   Investigation results: {investigation_count} entries")
            print(f"   Discussion history: {discussion_count} entries")

            # Discussion history should be capped at 5
            if discussion_count <= 5:
                print(f"   ✅ Discussion history properly capped")
                memory_ok = True
            else:
                print(f"   ❌ Discussion history not capped ({discussion_count} > 5)")
                memory_ok = False

            # Investigation results can grow but shouldn't be massive
            if investigation_count <= 10:
                print(f"   ✅ Investigation results reasonable")
            else:
                print(f"   ⚠️  Many investigation results ({investigation_count})")

            return memory_ok

        except Exception as e:
            print(f"❌ Memory test failed: {e}")
            return False

    def test_meta_commentary_frequency(self) -> bool:
        """Test how often meta-commentary appears"""
        print("\n🗣️  Testing Meta-Commentary Frequency...")
        print("-" * 60)

        meta_patterns = [
            "okay, let",
            "the user wants",
            "i need to",
            "my task is",
            "let me",
        ]

        tries = 5
        meta_count = 0

        for i in range(tries):
            try:
                self.agent.my_role = "villager"
                self.agent.my_player_id = 1
                self.agent.alive_players = [0, 1, 2, 3]
                self.agent.turn_count = i + 1

                obs = f"""Day breaks. Discuss for 3 rounds.
Player 0: Who is suspicious?
Player 2: Not sure yet."""

                response = self.agent(obs)
                response_lower = response.lower()

                has_meta = any(pattern in response_lower for pattern in meta_patterns)

                # Show the actual final response
                response_preview = response[:80] + "..." if len(response) > 80 else response

                if has_meta:
                    meta_count += 1
                    print(f"   Try {i+1}: ⚠️  Meta-commentary detected")
                    print(f"            '{response_preview}'")
                else:
                    print(f"   Try {i+1}: ✅ Clean response")
                    print(f"            Final output: '{response_preview}'")

            except Exception as e:
                print(f"   Try {i+1}: ❌ Error: {e}")

        meta_rate = meta_count / tries
        print(f"\n   Meta-commentary rate: {meta_count}/{tries} ({meta_rate*100:.0f}%)")

        if meta_rate <= 0.3:  # 30% or less is acceptable
            print(f"   ✅ Acceptable meta-commentary rate")
            return True
        else:
            print(f"   ⚠️  High meta-commentary rate (>{30}%)")
            return False

    def run_all_tests(self) -> dict:
        """Run all edge case tests"""
        print("=" * 70)
        print("🔍 COMPREHENSIVE EDGE CASE TEST SUITE - LOCAL LLM")
        print("=" * 70)
        print(f"\nModel: {self.model_name}")
        print(f"Quantization: {'4-bit' if self.use_4bit else 'Full precision'}")
        print("Testing: Every possible failure mode\n")

        # Initialize agent once
        self.initialize_agent()

        tests = [
            ("Invalid Observations", self.test_invalid_observations),
            ("Timeout Scenarios", self.test_timeout_scenarios),
            ("Consecutive Games (5)", self.test_consecutive_games_stability),
            ("Long Game (20 turns)", self.test_long_game_stability),
            ("Response Format", self.test_response_format_validation),
            ("Player Count Variations", self.test_player_count_variations),
            ("Role Reveal Safety", self.test_role_reveal_safety),
            ("Invalid Move Recovery", self.test_invalid_move_recovery),
            ("Memory Overflow", self.test_memory_overflow_protection),
            ("Meta-Commentary Rate", self.test_meta_commentary_frequency),
        ]

        for test_name, test_func in tests:
            try:
                print(f"\n{'='*70}")
                result = test_func()
                self.test_results[test_name] = result
            except Exception as e:
                print(f"\n❌ {test_name} crashed: {e}")
                import traceback
                traceback.print_exc()
                self.test_results[test_name] = False

        return self.test_results

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 70)
        print("📊 EDGE CASE TEST RESULTS - LOCAL LLM")
        print("=" * 70)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:.<45} {status}")

        print(f"\nSuccess Rate: {passed}/{total} ({passed/total*100:.1f}%)")

        if passed == total:
            print("\n🎉 ALL EDGE CASES HANDLED!")
            print("\n✅ Local Agent is production-ready")
            return True
        elif passed >= total * 0.8:
            print("\n⚠️  MOST EDGE CASES HANDLED (80%+)")
            print("Local Agent is deployable with known limitations")
            return True
        else:
            print("\n❌ TOO MANY EDGE CASE FAILURES")
            print("Not recommended for deployment")
            return False


def main():
    """Main test execution"""
    print(f"\n🔬 Starting Local LLM Edge Case Testing")
    print(f"Model: {MODEL_NAME}")
    print(f"Quantization: {'4-bit' if USE_4BIT else 'Full precision'}\n")
    print("⚠️  This will take 10-20 minutes (model loading + inference)...\n")

    tester = LocalEdgeCaseTester(MODEL_NAME, USE_4BIT)
    tester.run_all_tests()
    success = tester.print_summary()

    if success:
        print("\n✅ Local Agent ready for production!")
        sys.exit(0)
    else:
        print("\n⚠️  Review failures before deployment")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
