"""
Test Face Recognition API
Simple script to test face authentication
"""
import requests
import base64

# Configuration
FACE_API_URL = "http://localhost:8001/face-match"
MOBILE = "8688922852"


def test_face_authentication(mobile, image_path):
    """Test face authentication with image file"""
    print("\n" + "="*60)
    print("Testing Face Recognition API")
    print("="*60)
    print(f"Mobile: {mobile}")
    print(f"Image: {image_path}")
    print(f"API: {FACE_API_URL}")
    print("="*60 + "\n")
    
    try:
        # Read and encode image
        print("Reading image file...")
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_base64 = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}"
        
        print(f"Image encoded: {len(image_base64)} characters\n")
        
        # Make API call
        print("Calling Face API...")
        response = requests.post(FACE_API_URL, json={
            "mob": mobile,
            "capturedImg": image_base64
        })
        
        # Display result
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("✓ SUCCESS")
            print("="*60)
            print(f"Matched: {'YES ✓' if result['matched'] else 'NO ✗'}")
            print(f"Confidence: {result['confidence']}%")
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
        print(f"✗ Error: Image file not found - {image_path}")
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to Face API")
        print("Make sure the server is running: python face.py")
    except Exception as e:
        print(f"✗ Error: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Face Recognition API - Test Script")
    print("="*60)
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8001/health", timeout=2)
        if health.status_code == 200:
            print("✓ Face API is running")
        else:
            print("⚠ Face API health check failed")
    except:
        print("✗ Face API is NOT running")
        print("Please start it first: python face.py")
        exit(1)
    
    # Instructions
    print("\nTo test face authentication, call:")
    print("  test_face_authentication('8688922852', 'path/to/image.jpg')")
    print("\nExample:")
    print("  test_face_authentication('8688922852', 'test_image.jpg')")
    print("\n" + "="*60)
    
    # Uncomment to test with your image:
    # test_face_authentication(MOBILE, "your_image.jpg")
