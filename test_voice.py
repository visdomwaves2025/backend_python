"""
Test Voice Recognition API
Simple script to test voice authentication
"""
import requests
import base64

# Configuration
VOICE_API_URL = "http://localhost:8002/voice-match"
MOBILE = "8688922852"


def test_voice_authentication(mobile, audio_path):
    """Test voice authentication with audio file"""
    print("\n" + "="*60)
    print("Testing Voice Recognition API")
    print("="*60)
    print(f"Mobile: {mobile}")
    print(f"Audio: {audio_path}")
    print(f"API: {VOICE_API_URL}")
    print("="*60 + "\n")
    
    try:
        # Read and encode audio
        print("Reading audio file...")
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            
            # Determine mime type
            if audio_path.lower().endswith('.wav'):
                mime_type = "audio/wav"
            elif audio_path.lower().endswith('.mp3'):
                mime_type = "audio/mp3"
            elif audio_path.lower().endswith('.ogg'):
                mime_type = "audio/ogg"
            else:
                mime_type = "audio/wav"
            
            audio_base64 = f"data:{mime_type};base64,{base64.b64encode(audio_data).decode()}"
        
        print(f"Audio encoded: {len(audio_base64)} characters\n")
        
        # Make API call
        print("Calling Voice API...")
        response = requests.post(VOICE_API_URL, json={
            "mob": mobile,
            "capturedAudio": audio_base64
        })
        
        # Display result
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("✓ SUCCESS")
            print("="*60)
            print(f"Matched: {'YES ✓' if result['matched'] else 'NO ✗'}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Voice Quality: {result['voiceQuality'] * 100:.1f}%")
            print(f"Message: {result['message']}")
            
            if result.get('studentInfo'):
                info = result['studentInfo']
                print(f"\nStudent Details:")
                print(f"  Name: {info['name']}")
                print(f"  UID: {info['uid']}")
                print(f"  Class: {info['class']} - {info['section']}")
                print(f"  Company: {info['companyName']}")
            print("="*60)
        else:
            print(f"\n✗ ERROR ({response.status_code})")
            print(f"Details: {response.json()}")
            
    except FileNotFoundError:
        print(f"✗ Error: Audio file not found - {audio_path}")
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to Voice API")
        print("Make sure the server is running: python voice.py")
    except Exception as e:
        print(f"✗ Error: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Voice Recognition API - Test Script")
    print("="*60)
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8002/health", timeout=2)
        if health.status_code == 200:
            print("✓ Voice API is running")
        else:
            print("⚠ Voice API health check failed")
    except:
        print("✗ Voice API is NOT running")
        print("Please start it first: python voice.py")
        exit(1)
    
    # Instructions
    print("\nTo test voice authentication, call:")
    print("  test_voice_authentication('8688922852', 'path/to/audio.wav')")
    print("\nExample:")
    print("  test_voice_authentication('8688922852', 'test_audio.wav')")
    print("\nSupported formats: WAV, MP3, OGG")
    print("Recommended: 10 seconds, 16kHz, mono")
    print("\n" + "="*60)
    
    # Uncomment to test with your audio:
    # test_voice_authentication(MOBILE, "your_audio.wav")
