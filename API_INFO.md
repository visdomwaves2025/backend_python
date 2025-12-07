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
  "confidence": 95.5,
  "message": "Face matched successfully!",
  "studentInfo": {
    "name": "John Doe",
    "uid": "STU123",
    "class": "10",
    "section": "A",
    "companyName": "VW EduTech",
    "branchName": "Main Branch"
  }
}
```

### Other Endpoints
- `GET http://localhost:8001/health` - Health check
- `GET http://localhost:8001/docs` - API documentation

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
  "capturedAudio": "data:audio/wav;base64,UklGRiQAAABXQVZF..."
}
```

### Response
```json
{
  "matched": true,
  "confidence": 92.3,
  "voiceQuality": 0.85,
  "message": "Voice matched successfully!",
  "studentInfo": {
    "name": "John Doe",
    "uid": "STU123",
    "class": "10",
    "section": "A",
    "companyName": "VW EduTech",
    "branchName": "Main Branch"
  }
}
```

### Other Endpoints
- `GET http://localhost:8002/health` - Health check
- `GET http://localhost:8002/docs` - API documentation

---

## 3. External Backend API (Used by both Face & Voice APIs)

**URL:** `https://dev.gaitview.com:449/login/custListByMob`

### API Call
```
POST https://dev.gaitview.com:449/login/custListByMob
```

### Request Body
```json
{
  "mob": "8688922852",
  "reqType": "stu"
}
```

### Response Fields Used
- `imgString` - Base64 face image (for Face API)
- `voiceString` - Base64 voice audio (for Voice API)
- `name`, `uid`, `cls`, `section`, `compName`, `branchName` - Student info

---

## Where API Calls Are Made

### In face.py
```python
# Line ~90-100: get_student_data() function
url = "https://dev.gaitview.com:449/login/custListByMob"
response = requests.post(url, json={"mob": mobile, "reqType": "stu"})
```

### In voice.py
```python
# Line ~45-55: get_student_data() function
url = "https://dev.gaitview.com:449/login/custListByMob"
response = requests.post(url, json={"mob": mobile, "reqType": "stu"})
```

---

## Quick Test Commands

### Test Face API
```bash
python test_face.py
```

### Test Voice API
```bash
python test_voice.py
```

---

## Summary

| API | Port | Endpoint | Input Field | Output Extra |
|-----|------|----------|-------------|--------------|
| Face | 8001 | `/face-match` | `capturedImg` | - |
| Voice | 8002 | `/voice-match` | `capturedAudio` | `voiceQuality` |

Both APIs call the same external backend to fetch student data.
