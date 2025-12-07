"""
Face Recognition API
Compares captured face image with stored image from external API
Self-contained with all face processing utilities
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import cv2
import base64
import io
from PIL import Image
from typing import Dict, Tuple, List
import insightface
from insightface.app import FaceAnalysis
import warnings
warnings.filterwarnings('ignore')


app = FastAPI(title="Face Recognition API")


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class FaceMatchRequest(BaseModel):
    mob: str
    capturedImg: str


class FaceMatchResponse(BaseModel):
    matched: bool
    confidence: float = 0.0
    message: str
    studentInfo: dict = None
    faceQuality: float = 0.0


# Initialize face analysis
face_app = None

def initialize_face_app():
    """Initialize InsightFace app"""
    global face_app
    if face_app is None:
        try:
            face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("Face analysis model initialized successfully")
        except Exception as e:
            print(f"Error initializing face analysis: {e}")
            face_app = None
    return face_app


# ============================================================================
# FACE PROCESSING UTILITIES
# ============================================================================


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 image to numpy array"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL to OpenCV format
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")


def extract_face_embedding(image: np.ndarray) -> Dict:
    """Extract face embedding from image"""
    try:
        app = initialize_face_app()
        if app is None:
            return {
                'embedding': None,
                'quality': 0.0,
                'error': 'Face analysis model not initialized'
            }
        
        # Detect faces
        faces = app.get(image)
        
        if len(faces) == 0:
            return {
                'embedding': None,
                'quality': 0.0,
                'error': 'No face detected in image'
            }
        
        if len(faces) > 1:
            return {
                'embedding': None,
                'quality': 0.0,
                'error': 'Multiple faces detected. Please ensure only one face is visible'
            }
        
        face = faces[0]
        
        # Calculate face quality based on various factors
        quality = calculate_face_quality(face, image)
        
        if quality < 0.3:
            return {
                'embedding': None,
                'quality': quality,
                'error': f'Face quality too low: {round(quality * 100)}%'
            }
        
        # Get face embedding
        embedding = face.embedding
        
        return {
            'embedding': embedding.tolist(),
            'quality': quality,
            'bbox': face.bbox.tolist(),
            'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None
        }
        
    except Exception as e:
        return {
            'embedding': None,
            'quality': 0.0,
            'error': f'Face extraction failed: {str(e)}'
        }


def calculate_face_quality(face, image: np.ndarray) -> float:
    """Calculate face quality score (0-1)"""
    try:
        quality_score = 0.0
        
        # Face size quality (larger faces are better)
        bbox = face.bbox
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        face_area = face_width * face_height
        image_area = image.shape[0] * image.shape[1]
        size_ratio = face_area / image_area
        size_score = min(1.0, size_ratio * 10)  # Normalize to 0-1
        
        # Face detection confidence
        det_score = getattr(face, 'det_score', 0.8)
        
        # Pose quality (frontal faces are better)
        pose_score = 1.0
        if hasattr(face, 'pose'):
            # Penalize extreme poses
            yaw, pitch, roll = face.pose
            pose_penalty = (abs(yaw) + abs(pitch) + abs(roll)) / 90.0
            pose_score = max(0.0, 1.0 - pose_penalty)
        
        # Age factor (avoid very young or very old faces if available)
        age_score = 1.0
        if hasattr(face, 'age'):
            age = face.age
            if 18 <= age <= 65:
                age_score = 1.0
            else:
                age_score = 0.8
        
        # Combine all quality factors
        quality_score = (size_score * 0.3 + 
                        det_score * 0.4 + 
                        pose_score * 0.2 + 
                        age_score * 0.1)
        
        return max(0.0, min(1.0, quality_score))
        
    except Exception as e:
        print(f"Error calculating face quality: {e}")
        return 0.5  # Default quality


def compare_face_embeddings(embedding1: List[float], embedding2: List[float]) -> float:
    """Compare two face embeddings using cosine similarity"""
    try:
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have same dimensions")
        
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert to distance (0 = identical, 1 = completely different)
        distance = (1.0 - similarity) / 2.0
        
        return float(distance)
        
    except Exception as e:
        raise ValueError(f"Face comparison failed: {str(e)}")


def validate_face_embedding(embedding: List[float]) -> bool:
    """Validate face embedding"""
    if not isinstance(embedding, list) or len(embedding) != 512:
        return False
    
    if any(not isinstance(val, (int, float)) or np.isnan(val) for val in embedding):
        return False
    
    # Check if embedding has meaningful values
    embedding_array = np.array(embedding)
    if np.linalg.norm(embedding_array) < 0.1:
        return False
    
    return True


# ============================================================================
# API FUNCTIONS
# ============================================================================


def get_student_data(mobile: str) -> dict:
    """Fetch student data from external API"""
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
        
        return data["value"][0]
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching student data: {str(e)}")


def compare_faces(captured_image: np.ndarray, stored_image_base64: str) -> dict:
    """Compare two face images"""
    try:
        print("Extracting embedding from captured image...")
        captured_result = extract_face_embedding(captured_image)
        
        if 'error' in captured_result:
            return {
                "matched": False,
                "confidence": 0.0,
                "quality": captured_result.get('quality', 0.0),
                "error": captured_result['error']
            }
        
        if captured_result['quality'] < 0.4:
            return {
                "matched": False,
                "confidence": 0.0,
                "quality": captured_result['quality'],
                "error": f"Face quality too low: {round(captured_result['quality'] * 100)}%"
            }
        
        captured_embedding = captured_result['embedding']
        
        if not validate_face_embedding(captured_embedding):
            return {
                "matched": False,
                "confidence": 0.0,
                "quality": captured_result['quality'],
                "error": "Invalid face embedding"
            }
        
        print("Extracting embedding from stored image...")
        stored_image = base64_to_image(stored_image_base64)
        stored_result = extract_face_embedding(stored_image)
        
        if 'error' in stored_result:
            return {
                "matched": False,
                "confidence": 0.0,
                "quality": captured_result['quality'],
                "error": f"Stored image error: {stored_result['error']}"
            }
        
        stored_embedding = stored_result['embedding']
        
        if not validate_face_embedding(stored_embedding):
            return {
                "matched": False,
                "confidence": 0.0,
                "quality": captured_result['quality'],
                "error": "Invalid stored face embedding"
            }
        
        print("Comparing face embeddings...")
        distance = compare_face_embeddings(captured_embedding, stored_embedding)
        
        # Matching thresholds
        MATCH_THRESHOLD = 0.35  # Distance threshold for face matching
        MIN_CONFIDENCE = 75     # Minimum confidence percentage
        
        confidence = max(0, min(100, (1 - distance * 2) * 100))
        
        matched = (distance < MATCH_THRESHOLD and confidence >= MIN_CONFIDENCE)
        
        print(f"Face comparison - Distance: {distance:.4f}, Confidence: {confidence:.2f}%")
        
        return {
            "matched": matched,
            "confidence": round(confidence, 2),
            "quality": round(captured_result['quality'], 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face comparison failed: {str(e)}")


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint"""
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
    """Face matching endpoint"""
    if not request.mob or len(request.mob) < 10:
        raise HTTPException(status_code=400, detail="Invalid mobile number")
    
    if not request.capturedImg:
        raise HTTPException(status_code=400, detail="Captured image is required")
    
    try:
        # Step 1: Get student data from external API
        print(f"Fetching student data for mobile: {request.mob}")
        student_data = get_student_data(request.mob)
        
        # Extract stored image
        stored_image_base64 = student_data.get("imageString")
        if not stored_image_base64:
            raise HTTPException(status_code=404, detail="No face image found for this student")
        
        # Step 2: Convert captured image from base64
        print("Processing captured image...")
        captured_image = base64_to_image(request.capturedImg)
        
        # Step 3: Compare faces
        print("Comparing faces...")
        comparison_result = compare_faces(captured_image, stored_image_base64)
        
        # Check for errors in comparison
        if "error" in comparison_result:
            raise HTTPException(status_code=400, detail=comparison_result["error"])
        
        # Step 4: Prepare response
        matched = comparison_result["matched"]
        confidence = comparison_result.get("confidence", 0.0)
        quality = comparison_result.get("quality", 0.0)
        
        response = FaceMatchResponse(
            matched=matched,
            confidence=confidence,
            faceQuality=quality,
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
        
        print(f"Result: {response.message} (Confidence: {confidence}%, Quality: {quality})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
