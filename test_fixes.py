#!/usr/bin/env python3
"""
Test script to verify the fixes for phase detection and target extraction
"""

import sys
import os
sys.path.append('src')

from utils.streamlined_mafia_agent import StreamlinedMafiaAgent

def test_phase_detection():
    """Test phase detection scenarios that previously failed"""
    
    # Create agent for testing
    agent = StreamlinedMafiaAgent(modal_endpoint_url="test_url")
    agent.my_player_id = 0
    agent.my_role = "detective"
    agent.alive_players = [0, 2, 3, 4]
    agent.turn_count = 6
    
    print("=" * 60)
    print("TESTING PHASE DETECTION")
    print("=" * 60)
    
    # Test case 1: Turn 6 scenario (night phase with investigation)
    test1_observation = """[GAME] Player 1 was eliminated by vote.
[GAME] Night phase - choose one player to investigate: [2], [3], [4]"""
    
    detected_phase = agent._detect_current_phase(test1_observation)
    print(f"Test 1 - Night investigation phase:")
    print(f"  Input: {test1_observation}")
    print(f"  Detected: {detected_phase}")
    print(f"  Expected: night")
    print(f"  ✅ PASS" if detected_phase == "night" else f"  ❌ FAIL")
    
    # Test case 2: Voting phase
    test2_observation = """[GAME] Day breaks. Discuss for 3 rounds, then a vote will follow.
[GAME] Voting phase - submit one vote in format [X]. Valid: [0], [1], [2], [3], [4]"""
    
    detected_phase = agent._detect_current_phase(test2_observation)
    print(f"\nTest 2 - Voting phase:")
    print(f"  Input: {test2_observation}")
    print(f"  Detected: {detected_phase}")
    print(f"  Expected: voting")
    print(f"  ✅ PASS" if detected_phase == "voting" else f"  ❌ FAIL")
    
    # Test case 3: Error recovery
    test3_observation = """[GAME] Player 0 attempted an invalid move. Reason: Invalid investigation target. Please resubmit a valid move and remember to follow the game rules to avoid penalties."""
    
    detected_phase = agent._detect_current_phase(test3_observation)
    print(f"\nTest 3 - Error recovery:")
    print(f"  Input: {test3_observation}")
    print(f"  Detected: {detected_phase}")
    print(f"  Expected: night")
    print(f"  ✅ PASS" if detected_phase == "night" else f"  ❌ FAIL")

def test_valid_target_extraction():
    """Test valid target extraction scenarios that previously failed"""
    
    # Create agent for testing
    agent = StreamlinedMafiaAgent(modal_endpoint_url="test_url")
    agent.my_player_id = 0
    agent.alive_players = [0, 2, 3, 4]
    
    print("\n" + "=" * 60)
    print("TESTING VALID TARGET EXTRACTION")
    print("=" * 60)
    
    # Test case 1: Night investigation targets
    test1_observation = """[GAME] Player 1 was eliminated by vote.
[GAME] Night phase - choose one player to investigate: [2], [3], [4]"""
    
    extracted_targets = agent._extract_valid_targets_from_observation(test1_observation)
    print(f"Test 1 - Night investigation targets:")
    print(f"  Input: {test1_observation}")
    print(f"  Extracted: {extracted_targets}")
    print(f"  Expected: [2, 3, 4]")
    print(f"  ✅ PASS" if extracted_targets == [2, 3, 4] else f"  ❌ FAIL")
    
    # Test case 2: Voting targets
    test2_observation = """[GAME] Voting phase - submit one vote in format [X]. Valid: [0], [1], [2], [3], [4]"""
    
    extracted_targets = agent._extract_valid_targets_from_observation(test2_observation)
    print(f"\nTest 2 - Voting targets:")
    print(f"  Input: {test2_observation}")
    print(f"  Extracted: {extracted_targets}")
    print(f"  Expected: [0, 1, 2, 3, 4]")
    print(f"  ✅ PASS" if extracted_targets == [0, 1, 2, 3, 4] else f"  ❌ FAIL")
    
    # Test case 3: Night protection targets
    test3_observation = """[GAME] We are in the Night phase. Since you are the doctor, you can decide which player to save.
       Simply reply in the following format: '[Player X]' or '[X]'
       valid options: '[0]', '[1]', '[2]', '[4]', '[5]'"""
    
    extracted_targets = agent._extract_valid_targets_from_observation(test3_observation)
    print(f"\nTest 3 - Doctor protection targets:")
    print(f"  Input: {test3_observation}")
    print(f"  Extracted: {extracted_targets}")
    print(f"  Expected: [0, 1, 2, 4, 5]")
    print(f"  ✅ PASS" if extracted_targets == [0, 1, 2, 4, 5] else f"  ❌ FAIL")

def test_move_extraction():
    """Test move extraction with validation against valid targets"""
    
    # Create agent for testing
    agent = StreamlinedMafiaAgent(modal_endpoint_url="test_url")
    agent.my_player_id = 0
    agent.alive_players = [0, 2, 3, 4]
    
    print("\n" + "=" * 60)
    print("TESTING MOVE EXTRACTION & VALIDATION")
    print("=" * 60)
    
    # Test case 1: Valid target selection
    test_observation = """[GAME] Night phase - choose one player to investigate: [2], [3], [4]"""
    test_response = "[3]"
    
    extracted_move = agent._extract_move(test_response, test_observation)
    print(f"Test 1 - Valid target selection:")
    print(f"  LLM Response: {test_response}")
    print(f"  Valid Targets: [2, 3, 4]")
    print(f"  Extracted: {extracted_move}")
    print(f"  Expected: [3]")
    print(f"  ✅ PASS" if extracted_move == "[3]" else f"  ❌ FAIL")
    
    # Test case 2: Invalid target with fallback
    test_response = "[1]"  # Dead player
    
    extracted_move = agent._extract_move(test_response, test_observation)
    print(f"\nTest 2 - Invalid target with fallback:")
    print(f"  LLM Response: {test_response}")
    print(f"  Valid Targets: [2, 3, 4]")
    print(f"  Extracted: {extracted_move}")
    print(f"  Expected: [2] (fallback)")
    print(f"  ✅ PASS" if extracted_move == "[2]" else f"  ❌ FAIL")

def main():
    """Run all tests"""
    print("🧪 TESTING MAFIA AGENT FIXES")
    print("Testing the fixes for phase detection, target extraction, and dead player handling")
    
    test_phase_detection()
    test_valid_target_extraction()
    test_move_extraction()
    
    print("\n" + "=" * 60)
    print("🏆 TESTING COMPLETE")
    print("=" * 60)
    print("If all tests show ✅ PASS, the fixes are working correctly!")

if __name__ == "__main__":
    main()