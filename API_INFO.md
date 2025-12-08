# API Calls Information

## 1. Face Recognition API

**Server File:** `face.py`  
**Port:** 8001  
**Start Command:** `python face.py`

### API Call
```
POST http://localhost:8001/face-match
```

### Request Body
```json
{
  "mob": "8688922852",
  "capturedImg": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### Response
```json
{
  "matched": true,
  "confidence": 85.67,
  "message": "Face matched successfully!",
  "faceQuality": 0.78,
  "studentInfo": {
    "name": "John Doe",
    "uid": "STU001",
    "class": "10",
    "section": "A",
    "companyName": "VW EduTech",
    "branchName": "Main Branch"
  }
}
```

### Error Response
```json
{
  "matched": false,
  "confidence": 0.0,
  "message": "Face does not match",
  "faceQuality": 0.45,
  "studentInfo": null
}
```

---

## 2. Voice Recognition API

**Server File:** `voice.py`  
**Port:** 8002  
**Start Command:** `python voice.py`

### API Call
```
POST http://localhost:8002/voice-match
```

### Request Body
```json
{
  "mob": "8688922852",
  "capturedAudio": "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEA..."
}
```

### Response
```json
{
  "matched": true,
  "confidence": 82.45,
  "message": "Voice matched successfully!",
  "voiceQuality": 0.85,
  "studentInfo": {
    "name": "John Doe",
    "uid": "STU001",
    "class": "10",
    "section": "A",
    "companyName": "VW EduTech",
    "branchName": "Main Branch"
  }
}
```

### Error Response
```json
{
  "matched": false,
  "confidence": 0.0,
  "message": "Voice does not match",
  "voiceQuality": 0.35,
  "studentInfo": null
}
```

---

## 3. Health Check Endpoints

### Face API Health Check
```
GET http://localhost:8001/health
```

### Voice API Health Check
```
GET http://localhost:8002/health
```

### Response
```json
{
  "status": "healthy",
  "service": "Face Recognition API" // or "Voice Recognition API"
}
```

---

## 4. API Documentation

### Face API Documentation
- **Swagger UI:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

### Voice API Documentation
- **Swagger UI:** http://localhost:8002/docs
- **ReDoc:** http://localhost:8002/redoc

---

## 5. Setup Instructions

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the APIs

#### Start Face Recognition API
```bash
python face.py
```
Server will start on: http://localhost:8001

#### Start Voice Recognition API
```bash
python voice.py
```
Server will start on: http://localhost:8002

### Both APIs can run simultaneously on different ports.

---

## 6. External API Integration

Both APIs integrate with the external student database:
- **URL:** `https://dev.gaitview.com:449/login/custListByMob`
- **Method:** POST
- **Payload:** 
  ```json
  {
    "mob": "mobile_number",
    "reqType": "stu"
  }
  ```

---

## 7. Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request (Invalid input) |
| 404 | Student not found |
| 500 | Internal Server Error |

---

## 8. Quality Thresholds

### Face Recognition
- **Minimum Face Quality:** 40%
- **Minimum Confidence:** 75%
- **Distance Threshold:** 0.35

### Voice Recognition
- **Minimum Voice Quality:** 50%
- **Minimum Confidence:** 78%
- **Distance Threshold:** 0.26

---

## 9. Supported Formats

### Face Recognition
- **Input:** Base64 encoded images (JPEG, PNG)
- **Processing:** RGB format, single face detection

### Voice Recognition
- **Input:** Base64 encoded audio (WAV, MP3)
- **Processing:** 16kHz mono audio, voice activity detection

---

## 10. Security Features

### Face Recognition
- Single face validation
- Face quality assessment
- Pose validation
- Age verification

### Voice Recognition
- Voice activity detection
- Speaker biometric validation
- Pitch and spectral analysis
- Anti-spoofing measures
