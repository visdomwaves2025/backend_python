"""
Test script for Voice Recognition API
Tests the voice matching functionality with sample data
"""
import requests
import json
import base64
import numpy as np
import librosa
import soundfile as sf
import io
from typing import Dict, Any


# API Configuration
API_BASE_URL = "http://localhost:8002"
VOICE_MATCH_ENDPOINT = f"{API_BASE_URL}/voice-match"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


def generate_test_audio(duration: float = 3.0, sample_rate: int = 16000) -> str:
    """Generate a test audio signal and convert to base64"""
    try:
        # Generate a simple test audio with voice-like characteristics
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a complex waveform that mimics voice characteristics
        # Fundamental frequency around 150 Hz (typical male voice)
        f0 = 150
        
        # Add harmonics to make it more voice-like
        audio = (np.sin(2 * np.pi * f0 * t) * 0.3 +
                np.sin(2 * np.pi * f0 * 2 * t) * 0.2 +
                np.sin(2 * np.pi * f0 * 3 * t) * 0.1 +
                np.sin(2 * np.pi * f0 * 4 * t) * 0.05)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.02, len(audio))
        audio = audio + noise
        
        # Apply envelope to make it more speech-like
        envelope = np.exp(-t * 0.5) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        audio = audio * envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        audio_bytes = buffer.getvalue()
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return f"data:audio/wav;base64,{audio_base64}"
        
    except Exception as e:
        print(f"Error generating test audio: {e}")
        return None


def test_health_check() -> bool:
    """Test the health check endpoint"""
    try:
        print("Testing health check endpoint...")
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_voice_match_with_test_data() -> Dict[str, Any]:
    """Test voice matching with generated test data"""
    try:
        print("\nTesting voice match with generated test audio...")
        
        # Generate test audio
        test_audio = generate_test_audio()
        if not test_audio:
            return {"success": False, "error": "Failed to generate test audio"}
        
        # Test payload
        payload = {
            "mob": "1234567890",  # Test mobile number
            "capturedAudio": test_audio
        }
        
        print(f"Sending request to: {VOICE_MATCH_ENDPOINT}")
        print(f"Mobile: {payload['mob']}")
        print(f"Audio data length: {len(test_audio)} characters")
        
        response = requests.post(VOICE_MATCH_ENDPOINT, json=payload, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice match request successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            return {"success": True, "data": result}
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            print(f"❌ Voice match failed: {response.status_code}")
            print(f"Error: {error_data}")
            return {"success": False, "error": error_data}
            
    except Exception as e:
        print(f"❌ Voice match test error: {e}")
        return {"success": False, "error": str(e)}


def test_voice_match_with_real_mobile() -> Dict[str, Any]:
    """Test voice matching with a real mobile number (if available)"""
    try:
        print("\nTesting voice match with real mobile number...")
        
        # You can replace this with a real mobile number that exists in your database
        real_mobile = input("Enter a real mobile number to test (or press Enter to skip): ").strip()
        
        if not real_mobile:
            print("Skipping real mobile test...")
            return {"success": False, "error": "No mobile number provided"}
        
        # Generate test audio
        test_audio = generate_test_audio()
        if not test_audio:
            return {"success": False, "error": "Failed to generate test audio"}
        
        payload = {
            "mob": real_mobile,
            "capturedAudio": test_audio
        }
        
        print(f"Testing with mobile: {real_mobile}")
        
        response = requests.post(VOICE_MATCH_ENDPOINT, json=payload, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Real mobile test successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            return {"success": True, "data": result}
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            print(f"❌ Real mobile test failed: {response.status_code}")
            print(f"Error: {error_data}")
            return {"success": False, "error": error_data}
            
    except Exception as e:
        print(f"❌ Real mobile test error: {e}")
        return {"success": False, "error": str(e)}


def test_invalid_requests():
    """Test various invalid request scenarios"""
    print("\nTesting invalid request scenarios...")
    
    test_cases = [
        {
            "name": "Empty mobile number",
            "payload": {"mob": "", "capturedAudio": "data:audio/wav;base64,test"},
            "expected_status": 400
        },
        {
            "name": "Invalid mobile number",
            "payload": {"mob": "123", "capturedAudio": "data:audio/wav;base64,test"},
            "expected_status": 400
        },
        {
            "name": "Missing captured audio",
            "payload": {"mob": "1234567890", "capturedAudio": ""},
            "expected_status": 400
        },
        {
            "name": "Invalid audio format",
            "payload": {"mob": "1234567890", "capturedAudio": "invalid_audio_data"},
            "expected_status": 400
        }
    ]
    
    for test_case in test_cases:
        try:
            print(f"\nTesting: {test_case['name']}")
            response = requests.post(VOICE_MATCH_ENDPOINT, json=test_case['payload'], timeout=10)
            
            if response.status_code == test_case['expected_status']:
                print(f"✅ {test_case['name']} - Expected status {test_case['expected_status']}")
            else:
                print(f"❌ {test_case['name']} - Expected {test_case['expected_status']}, got {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test_case['name']} - Error: {e}")


def main():
    """Main test function"""
    print("=" * 60)
    print("VOICE RECOGNITION API TEST SUITE")
    print("=" * 60)
    
    # Test 1: Health Check
    health_ok = test_health_check()
    
    if not health_ok:
        print("\n❌ API is not running or not healthy. Please start the voice API first:")
        print("python voice.py")
        return
    
    # Test 2: Voice match with test data
    test_voice_match_with_test_data()
    
    # Test 3: Voice match with real mobile (optional)
    test_voice_match_with_real_mobile()
    
    # Test 4: Invalid requests
    test_invalid_requests()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
