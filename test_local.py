"""
Quick test script for local development
"""

import requests
import json
import sys
from typing import Optional

BASE_URL = "http://localhost:8000"

def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_detailed_health():
    """Test detailed health endpoint"""
    print("\n🔍 Testing detailed health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chat_history(ub_id: int):
    """Test chat history endpoint"""
    print(f"\n🔍 Testing chat history for UB ID: {ub_id}...")
    try:
        response = requests.get(f"{BASE_URL}/chat/{ub_id}/history")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['count']} messages")
            
            if data['messages']:
                print("\n📝 Message history:")
                for i, msg in enumerate(data['messages'][:5], 1):  # Show first 5
                    role = msg.get('role', 'unknown')
                    if role == 'user':
                        content = msg.get('user_content', {}).get('text', '')
                    else:
                        ai_content = msg.get('ai_content', [])
                        content = ai_content[0].get('text', '') if ai_content else ''
                    
                    print(f"\n  {i}. [{role.upper()}]")
                    print(f"     {content[:100]}{'...' if len(content) > 100 else ''}")
                
                if data['count'] > 5:
                    print(f"\n  ... and {data['count'] - 5} more messages")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_send_message(ub_id: int, message: str):
    """Test sending a message"""
    print(f"\n🔍 Testing send message to UB ID: {ub_id}...")
    print(f"📤 Message: {message}")
    try:
        response = requests.post(
            f"{BASE_URL}/chat/message",
            json={
                "ub_id": ub_id,
                "content": message
            },
            timeout=60  # AI responses can take time
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response received:")
            print(f"\n📥 AI Response:")
            print(f"   Type: {data.get('type')}")
            print(f"   Text: {data.get('text')}")
            if data.get('additional'):
                print(f"   Additional: {json.dumps(data.get('additional'), indent=2)}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_evaluation(ub_id: int):
    """Test evaluation endpoint"""
    print(f"\n🔍 Testing evaluation for UB ID: {ub_id}...")
    try:
        response = requests.post(
            f"{BASE_URL}/chat/{ub_id}/evaluate",
            timeout=60
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Evaluation completed:")
            print(f"\n📊 Results:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def interactive_mode():
    """Interactive testing mode"""
    print_section("🎮 Interactive Testing Mode")
    print("\nEnter UB ID (chat session ID) to test with real data")
    print("Or press Enter to skip interactive tests")
    
    ub_id_input = input("\nUB ID: ").strip()
    
    if not ub_id_input:
        print("⏭️  Skipping interactive tests")
        return
    
    try:
        ub_id = int(ub_id_input)
    except ValueError:
        print("❌ Invalid UB ID")
        return
    
    # Test history
    if test_chat_history(ub_id):
        # Ask if want to send message
        print("\n" + "-" * 60)
        send = input("\nSend a test message? (y/n): ").strip().lower()
        
        if send == 'y':
            message = input("Your message: ").strip()
            if message:
                test_send_message(ub_id, message)
        
        # Ask about evaluation
        print("\n" + "-" * 60)
        eval_test = input("\nRun evaluation on this chat? (y/n): ").strip().lower()
        
        if eval_test == 'y':
            test_evaluation(ub_id)

def main():
    """Main test function"""
    print_section("EdTech AI Platform - Local Testing")
    
    # Check if server is running
    print("\n🔌 Checking if server is running...")
    try:
        requests.get(f"{BASE_URL}/", timeout=2)
        print("✅ Server is running!")
    except:
        print("❌ Server is not running!")
        print(f"\nPlease start the server first:")
        print("  python main.py")
        sys.exit(1)
    
    # Basic health checks
    print_section("Basic Health Checks")
    test_health()
    test_detailed_health()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        ub_id = int(sys.argv[1])
        print_section(f"Testing with UB ID: {ub_id}")
        test_chat_history(ub_id)
        
        if len(sys.argv) > 2:
            message = " ".join(sys.argv[2:])
            test_send_message(ub_id, message)
    else:
        # Interactive mode
        interactive_mode()
    
    print_section("✅ Tests completed!")
    print("\nUsage tips:")
    print("  python test_local.py                    # Interactive mode")
    print("  python test_local.py 12518              # Test specific chat")
    print("  python test_local.py 12518 'Hello AI'   # Send message")

if __name__ == "__main__":
    main()