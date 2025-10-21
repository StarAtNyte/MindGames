#!/usr/bin/env python3
"""
Modal Endpoint Warmup Script
Wakes up the Modal endpoint before games to eliminate cold start delays
"""

import requests
import time
import sys

ENDPOINT_URL = "https://nitizkhanal000--enhanced-secretmafia-fastapi-app.modal.run"

def warmup_endpoint():
    """Warm up the Modal endpoint to avoid cold starts during games"""

    print("\n" + "=" * 70)
    print("MODAL ENDPOINT WARMUP")
    print("=" * 70)
    print(f"Endpoint: {ENDPOINT_URL}")
    print("=" * 70)

    # Step 1: Health check
    print("\n[1/3] Checking endpoint health...")
    try:
        response = requests.get(
            f"{ENDPOINT_URL}/health",
            timeout=120  # First request may take 30-60s
        )

        if response.status_code == 200:
            print("✓ Endpoint is healthy")
            data = response.json()
            print(f"  Model: {data.get('model', 'Unknown')}")
            print(f"  Status: {data.get('status', 'Unknown')}")
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("⚠️  Health check timed out (endpoint may be cold)")
        print("  Continuing with warmup requests...")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

    # Step 2: Warmup with a simple request
    print("\n[2/3] Sending warmup request (may take 30-60s on cold start)...")
    start_time = time.time()

    warmup_payload = {
        "observation": "Hello. Respond with 'Ready'.",
        "phase": "discussion"
    }

    try:
        response = requests.post(
            f"{ENDPOINT_URL}/generate",
            json=warmup_payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"✓ Warmup successful ({elapsed:.1f}s)")
            result = response.json()
            print(f"  Response: {result.get('response', 'No response')[:50]}")
        else:
            print(f"⚠️  Warmup request failed: {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Warmup timed out after 120s")
        print("  The endpoint may be experiencing issues")
        return False
    except Exception as e:
        print(f"❌ Warmup failed: {e}")
        return False

    # Step 3: Test with a game-like scenario
    print("\n[3/3] Testing with game scenario...")
    start_time = time.time()

    game_payload = {
        "observation": """Day 1. Discuss for 3 rounds.
Player 0: Who is suspicious?
Player 2: Not sure yet.""",
        "phase": "discussion"
    }

    try:
        response = requests.post(
            f"{ENDPOINT_URL}/generate",
            json=game_payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # Should be fast now
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"✓ Game scenario test successful ({elapsed:.1f}s)")
            result = response.json()
            response_text = result.get('response', 'No response')
            print(f"  Response preview: {response_text[:80]}...")
        else:
            print(f"⚠️  Game test failed: {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Game test timed out")
        return False
    except Exception as e:
        print(f"❌ Game test failed: {e}")
        return False

    # Success!
    print("\n" + "=" * 70)
    print("✅ ENDPOINT READY FOR GAMES")
    print("=" * 70)
    print("\nExpected response times:")
    print("  • First turn: ~5-10s")
    print("  • Subsequent turns: ~2-5s")
    print("\nYou can now run games without cold start delays:")
    print("  python run_match.py")
    print("  python src/online_play.py")
    print("=" * 70)

    return True

def main():
    """Main entry point"""
    print("\n🔥 Warming up Modal endpoint...")
    print("This will take 30-90 seconds on first run\n")

    try:
        success = warmup_endpoint()

        if success:
            sys.exit(0)
        else:
            print("\n⚠️  Warmup completed with warnings")
            print("The endpoint may still work, but could be slower")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Warmup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
