"""
Test script for Face Recognition API
Tests the face matching functionality with sample data
"""
import requests
import json
import base64
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io
from typing import Dict, Any


# API Configuration
API_BASE_URL = "http://localhost:8001"
FACE_MATCH_ENDPOINT = f"{API_BASE_URL}/face-match"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


def generate_test_face_image(width: int = 400, height: int = 400) -> str:
    """Generate a simple test face image and convert to base64"""
    try:
        # Create a simple face-like image using PIL
        image = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(image)
        
        # Draw a simple face
        # Face outline (circle)
        face_center = (width // 2, height // 2)
        face_radius = min(width, height) // 3
        draw.ellipse([
            face_center[0] - face_radius,
            face_center[1] - face_radius,
            face_center[0] + face_radius,
            face_center[1] + face_radius
        ], fill='peachpuff', outline='black', width=2)
        
        # Eyes
        eye_y = face_center[1] - face_radius // 3
        left_eye_x = face_center[0] - face_radius // 3
        right_eye_x = face_center[0] + face_radius // 3
        eye_radius = face_radius // 8
        
        # Left eye
        draw.ellipse([
            left_eye_x - eye_radius,
            eye_y - eye_radius,
            left_eye_x + eye_radius,
            eye_y + eye_radius
        ], fill='white', outline='black', width=1)
        
        # Left pupil
        draw.ellipse([
            left_eye_x - eye_radius // 2,
            eye_y - eye_radius // 2,
            left_eye_x + eye_radius // 2,
            eye_y + eye_radius // 2
        ], fill='black')
        
        # Right eye
        draw.ellipse([
            right_eye_x - eye_radius,
            eye_y - eye_radius,
            right_eye_x + eye_radius,
            eye_y + eye_radius
        ], fill='white', outline='black', width=1)
        
        # Right pupil
        draw.ellipse([
            right_eye_x - eye_radius // 2,
            eye_y - eye_radius // 2,
            right_eye_x + eye_radius // 2,
            eye_y + eye_radius // 2
        ], fill='black')
        
        # Nose
        nose_x = face_center[0]
        nose_y = face_center[1]
        nose_width = face_radius // 6
        nose_height = face_radius // 4
        draw.ellipse([
            nose_x - nose_width,
            nose_y - nose_height // 2,
            nose_x + nose_width,
            nose_y + nose_height // 2
        ], fill='rosybrown', outline='black', width=1)
        
        # Mouth
        mouth_y = face_center[1] + face_radius // 3
        mouth_width = face_radius // 2
        mouth_height = face_radius // 8
        draw.ellipse([
            face_center[0] - mouth_width,
            mouth_y - mouth_height,
            face_center[0] + mouth_width,
            mouth_y + mouth_height
        ], fill='red', outline='black', width=1)
        
        # Convert PIL image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
        
    except Exception as e:
        print(f"Error generating test image: {e}")
        return None


def generate_realistic_test_image() -> str:
    """Generate a more realistic test image with gradients"""
    try:
        # Create a more realistic face image
        width, height = 300, 400
        
        # Create base image with gradient background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            for x in range(width):
                image[y, x] = [200 - y//4, 220 - y//5, 240 - y//6]
        
        # Add face oval
        center = (width // 2, height // 2)
        axes = (width // 3, height // 2 - 20)
        cv2.ellipse(image, center, axes, 0, 0, 360, (220, 180, 140), -1)
        cv2.ellipse(image, center, axes, 0, 0, 360, (0, 0, 0), 2)
        
        # Add eyes
        left_eye = (center[0] - 40, center[1] - 40)
        right_eye = (center[0] + 40, center[1] - 40)
        cv2.circle(image, left_eye, 15, (255, 255, 255), -1)
        cv2.circle(image, left_eye, 8, (0, 0, 0), -1)
        cv2.circle(image, right_eye, 15, (255, 255, 255), -1)
        cv2.circle(image, right_eye, 8, (0, 0, 0), -1)
        
        # Add nose
        nose_points = np.array([
            [center[0] - 8, center[1] - 10],
            [center[0] + 8, center[1] - 10],
            [center[0], center[1] + 15]
        ], np.int32)
        cv2.fillPoly(image, [nose_points], (200, 160, 120))
        
        # Add mouth
        mouth_center = (center[0], center[1] + 50)
        cv2.ellipse(image, mouth_center, (25, 10), 0, 0, 180, (150, 50, 50), -1)
        
        # Convert to PIL and then to base64
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        image_bytes = buffer.getvalue()
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
        
    except Exception as e:
        print(f"Error generating realistic test image: {e}")
        return generate_test_face_image()  # Fallback to simple image


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


def test_face_match_with_test_data() -> Dict[str, Any]:
    """Test face matching with generated test data"""
    try:
        print("\nTesting face match with generated test image...")
        
        # Generate test image
        test_image = generate_realistic_test_image()
        if not test_image:
            return {"success": False, "error": "Failed to generate test image"}
        
        # Test payload
        payload = {
            "mob": "1234567890",  # Test mobile number
            "capturedImg": test_image
        }
        
        print(f"Sending request to: {FACE_MATCH_ENDPOINT}")
        print(f"Mobile: {payload['mob']}")
        print(f"Image data length: {len(test_image)} characters")
        
        response = requests.post(FACE_MATCH_ENDPOINT, json=payload, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Face match request successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            return {"success": True, "data": result}
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            print(f"❌ Face match failed: {response.status_code}")
            print(f"Error: {error_data}")
            return {"success": False, "error": error_data}
            
    except Exception as e:
        print(f"❌ Face match test error: {e}")
        return {"success": False, "error": str(e)}


def test_face_match_with_real_mobile() -> Dict[str, Any]:
    """Test face matching with a real mobile number (if available)"""
    try:
        print("\nTesting face match with real mobile number...")
        
        # You can replace this with a real mobile number that exists in your database
        real_mobile = input("Enter a real mobile number to test (or press Enter to skip): ").strip()
        
        if not real_mobile:
            print("Skipping real mobile test...")
            return {"success": False, "error": "No mobile number provided"}
        
        # Generate test image
        test_image = generate_realistic_test_image()
        if not test_image:
            return {"success": False, "error": "Failed to generate test image"}
        
        payload = {
            "mob": real_mobile,
            "capturedImg": test_image
        }
        
        print(f"Testing with mobile: {real_mobile}")
        
        response = requests.post(FACE_MATCH_ENDPOINT, json=payload, timeout=30)
        
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
            "payload": {"mob": "", "capturedImg": "data:image/jpeg;base64,test"},
            "expected_status": 400
        },
        {
            "name": "Invalid mobile number",
            "payload": {"mob": "123", "capturedImg": "data:image/jpeg;base64,test"},
            "expected_status": 400
        },
        {
            "name": "Missing captured image",
            "payload": {"mob": "1234567890", "capturedImg": ""},
            "expected_status": 400
        },
        {
            "name": "Invalid image format",
            "payload": {"mob": "1234567890", "capturedImg": "invalid_image_data"},
            "expected_status": 400
        }
    ]
    
    for test_case in test_cases:
        try:
            print(f"\nTesting: {test_case['name']}")
            response = requests.post(FACE_MATCH_ENDPOINT, json=test_case['payload'], timeout=10)
            
            if response.status_code == test_case['expected_status']:
                print(f"✅ {test_case['name']} - Expected status {test_case['expected_status']}")
            else:
                print(f"❌ {test_case['name']} - Expected {test_case['expected_status']}, got {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test_case['name']} - Error: {e}")


def test_image_formats():
    """Test different image formats and sizes"""
    print("\nTesting different image formats...")
    
    try:
        # Test with simple image
        simple_image = generate_test_face_image(200, 200)
        if simple_image:
            payload = {"mob": "1234567890", "capturedImg": simple_image}
            response = requests.post(FACE_MATCH_ENDPOINT, json=payload, timeout=15)
            print(f"Simple image test: {response.status_code}")
        
        # Test with larger image
        large_image = generate_test_face_image(800, 600)
        if large_image:
            payload = {"mob": "1234567890", "capturedImg": large_image}
            response = requests.post(FACE_MATCH_ENDPOINT, json=payload, timeout=15)
            print(f"Large image test: {response.status_code}")
            
    except Exception as e:
        print(f"Image format test error: {e}")


def main():
    """Main test function"""
    print("=" * 60)
    print("FACE RECOGNITION API TEST SUITE")
    print("=" * 60)
    
    # Test 1: Health Check
    health_ok = test_health_check()
    
    if not health_ok:
        print("\n❌ API is not running or not healthy. Please start the face API first:")
        print("python face.py")
        return
    
    # Test 2: Face match with test data
    test_face_match_with_test_data()
    
    # Test 3: Face match with real mobile (optional)
    test_face_match_with_real_mobile()
    
    # Test 4: Invalid requests
    test_invalid_requests()
    
    # Test 5: Different image formats
    test_image_formats()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
    print("\nNOTE: The generated test images are simple synthetic faces.")
    print("For real testing, use actual face photos.")
    print("InsightFace models work best with real human faces.")


if __name__ == "__main__":
    main()
