"""
Face Recognition API
Compares captured face image with stored image from external API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import base64
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import io
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Face Recognition API")

# Initialize InsightFace model (ArcFace)
face_app = FaceAnalysis('antelope')  # Using antelope model for version 0.2.1
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 for CPU


# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class FaceMatchRequest(BaseModel):
    mob: str
    capturedImg: str  # base64 encoded image

# Response model
class FaceMatchResponse(BaseModel):
    matched: bool
    confidence: float = 0.0
    message: str
    studentInfo: dict = None

def base64_to_cv2_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV image
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert PIL Image to OpenCV format (BGR)
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        return img_cv2
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def extract_face_embedding(img: np.ndarray) -> np.ndarray:
    """
    Extract face embedding using InsightFace ArcFace model
    Returns: 512-dimensional face embedding
    """
    try:
        # Detect and analyze faces
        faces = face_app.get(img)
        
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        # Get the first (largest) face
        face = faces[0]
        
        # Return face embedding (512-dimensional vector)
        return face.embedding
    except Exception as e:
        raise ValueError(f"Face extraction failed: {str(e)}")

def get_student_data(mobile: str) -> dict:
    """
    Fetch student data from external API
    """
    try:
        url = "https://dev.gaitview.com:449/login/custListByMob"
        payload = {
            "mob": mobile,
            "reqType": "stu"
        }
        
        response = requests.post(url, json=payload, verify=False, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("error"):
            raise HTTPException(status_code=404, detail=data.get("errorMessage", "Student not found"))
        
        if not data.get("value") or len(data["value"]) == 0:
            raise HTTPException(status_code=404, detail="No student found with this mobile number")
        
        return data["value"][0]  # Return first student record
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching student data: {str(e)}")

def compare_faces(captured_img: np.ndarray, stored_img_base64: str) -> dict:
    """
    Compare two face images using InsightFace ArcFace
    Returns: dict with 'matched' (bool) and 'confidence' (float)
    """
    try:
        # Convert stored image from base64
        stored_img = base64_to_cv2_image(stored_img_base64)
        
        # Extract face embeddings from both images
        print("Extracting face embedding from captured image...")
        captured_embedding = extract_face_embedding(captured_img)
        
        print("Extracting face embedding from stored image...")
        stored_embedding = extract_face_embedding(stored_img)
        
        # Calculate cosine similarity between embeddings
        # Normalize embeddings
        captured_norm = captured_embedding / np.linalg.norm(captured_embedding)
        stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
        
        # Cosine similarity
        similarity = np.dot(captured_norm, stored_norm)
        
        # Convert similarity to distance (1 - similarity)
        distance = 1.0 - similarity
        
        # Threshold for ArcFace (typical range: 0.4-0.6 for distance, lower is better)
        THRESHOLD = 0.5  # If distance < threshold, faces match
        
        # Calculate confidence (0-100%)
        # Lower distance = higher confidence
        confidence = max(0, min(100, (1 - distance / THRESHOLD) * 100))
        
        # Determine if faces match
        matched = distance < THRESHOLD and confidence >= 60
        
        print(f"Similarity: {similarity:.4f}, Distance: {distance:.4f}, Confidence: {confidence:.2f}%")
        
        return {
            "matched": matched,
            "confidence": round(confidence, 2),
            "similarity": round(float(similarity), 4),
            "distance": round(float(distance), 4)
        }
            
    except ValueError as e:
        # If face not detected, return not matched
        return {
            "matched": False,
            "confidence": 0.0,
            "error": str(e)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face comparison failed: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint - API info
    """
    return {
        "message": "VW EduTech Face Recognition API",
        "version": "1.0",
        "endpoints": {
            "face-match": "/face-match (POST)",
            "health": "/health (GET)"
        }
    }

@app.post("/face-match", response_model=FaceMatchResponse)
async def face_match(request: FaceMatchRequest):
    """
    Face matching endpoint
    
    Request Body:
    {
        "mob": "8688922852",
        "capturedImg": "data:image/jpeg;base64,/9j/4AAQ..."
    }
    
    Response:
    {
        "matched": true/false,
        "confidence": 95.5,
        "message": "Face matched successfully",
        "studentInfo": {...}
    }
    """
    # Validate mobile number
    if not request.mob or len(request.mob) < 10:
        raise HTTPException(status_code=400, detail="Invalid mobile number")
    
    # Validate captured image
    if not request.capturedImg:
        raise HTTPException(status_code=400, detail="Captured image is required")
    
    try:
        # Step 1: Get student data from external API
        print(f"Fetching student data for mobile: {request.mob}")
        student_data = get_student_data(request.mob)
        
        # Extract stored image
        stored_img_base64 = student_data.get("imgString")
        if not stored_img_base64:
            raise HTTPException(status_code=404, detail="No face image found for this student")
        
        # Step 2: Convert captured image from base64
        print("Processing captured image...")
        captured_img = base64_to_cv2_image(request.capturedImg)
        
        # Step 3: Compare faces
        print("Comparing faces...")
        comparison_result = compare_faces(captured_img, stored_img_base64)
        
        # Step 4: Prepare response
        matched = comparison_result["matched"]
        confidence = comparison_result.get("confidence", 0.0)
        
        response = FaceMatchResponse(
            matched=matched,
            confidence=confidence,
            message="Face matched successfully!" if matched else "Face does not match",
            studentInfo={
                "name": student_data.get("name", "").strip(),
                "uid": student_data.get("uid"),
                "class": student_data.get("cls"),
                "section": student_data.get("section"),
                "companyName": student_data.get("compName"),
                "branchName": student_data.get("branchName")
            } if matched else None
        )
        
        print(f"Result: {response.message} (Confidence: {confidence}%)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Face Recognition API"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Face Recognition API...")
    print("Server: http://localhost:8001")
    print("Endpoints:")
    print("  - POST /face-match - Face recognition")
    print("  - GET /health - Health check")
    print("API Docs: http://localhost:8001/docs")
    print("Interactive Docs: http://localhost:8001/redoc")
    uvicorn.run(app, host="127.0.0.1", port=8001)
