"""
Voice Recognition API
Compares captured voice audio with stored audio from external API
Self-contained with all voice processing utilities
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
import librosa
import io
import base64
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


app = FastAPI(title="Voice Recognition API")


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class VoiceMatchRequest(BaseModel):
    mob: str
    capturedAudio: str


class VoiceMatchResponse(BaseModel):
    matched: bool
    confidence: float = 0.0
    message: str
    studentInfo: dict = None
    voiceQuality: float = 0.0



# ============================================================================
# VOICE PROCESSING UTILITIES
# ============================================================================


def extract_voice_descriptor_from_audio(audio_data: np.ndarray, sample_rate: int = 16000, target_dimensions: int = 256) -> Dict:
    """Extract 256-dimensional voice descriptor from audio"""
    try:
        frame_size = 512
        hop_size = 256
        num_frames = (len(audio_data) - frame_size) // hop_size
        
        if num_frames < 10:
            return create_empty_descriptor(target_dimensions)
        
        features = {
            'mfcc': [],
            'pitch': [],
            'spectral_centroid': [],
            'spectral_rolloff': [],
            'spectral_flux': [],
            'zcr': [],
            'energy': [],
            'chroma': []
        }
        
        # MFCC features (13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, 
                                     n_fft=frame_size, hop_length=hop_size)
        features['mfcc'] = mfcc.T.flatten().tolist()
        
        # Pitch extraction using pyin algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, fmin=80, fmax=400, 
                                                      sr=sample_rate, frame_length=frame_size, 
                                                      hop_length=hop_size)
        features['pitch'] = np.nan_to_num(f0, nan=0.0).tolist()
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, 
                                                                n_fft=frame_size, hop_length=hop_size)
        features['spectral_centroid'] = spectral_centroids[0].tolist()
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate, 
                                                             n_fft=frame_size, hop_length=hop_size)
        features['spectral_rolloff'] = spectral_rolloff[0].tolist()
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=frame_size, 
                                                  hop_length=hop_size)
        features['zcr'] = zcr[0].tolist()
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_size, hop_length=hop_size)
        features['energy'] = rms[0].tolist()
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, 
                                              n_fft=frame_size, hop_length=hop_size)
        features['chroma'] = chroma.T.flatten().tolist()
        
        # Spectral flux
        S = np.abs(librosa.stft(audio_data, n_fft=frame_size, hop_length=hop_size))
        spectral_flux = []
        for i in range(1, S.shape[1]):
            flux = np.sqrt(np.sum((S[:, i] - S[:, i-1]) ** 2))
            spectral_flux.append(flux)
        features['spectral_flux'] = spectral_flux
        
        # Security checks
        voiced_pitches = [p for p in features['pitch'] if p > 0]
        voiced_ratio = len(voiced_pitches) / len(features['pitch']) if len(features['pitch']) > 0 else 0
        
        if voiced_ratio < 0.3:
            return create_empty_descriptor(target_dimensions)
        
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        if rms_energy < 0.005:
            return create_empty_descriptor(target_dimensions)
        
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 0.02:
            return create_empty_descriptor(target_dimensions)
        
        # Create descriptor
        descriptor = create_voice_descriptor(features, target_dimensions)
        
        descriptor_variance = np.var(descriptor)
        if descriptor_variance < 0.001:
            return create_empty_descriptor(target_dimensions)
        
        spectral_energy = np.mean(features['spectral_centroid'])
        if spectral_energy < 30:
            return create_empty_descriptor(target_dimensions)
        
        # Calculate quality
        quality = calculate_voice_quality(features, voiced_ratio)
        
        avg_pitch = np.mean(voiced_pitches) if voiced_pitches else 0
        pitch_variance = np.var(voiced_pitches) if voiced_pitches else 0
        
        return {
            'descriptor': descriptor,
            'quality': quality,
            'metadata': {
                'avgPitch': float(avg_pitch),
                'pitchVariance': float(pitch_variance),
                'voicedFrames': len(voiced_pitches),
                'totalFrames': len(features['pitch'])
            }
        }
        
    except Exception as e:
        print(f"Error extracting voice descriptor: {e}")
        return create_empty_descriptor(target_dimensions)



def create_voice_descriptor(features: Dict, target_dimensions: int) -> List[float]:
    """Create 256-dimensional voice descriptor"""
    descriptor = []
    
    # MFCC stats (26 values: 13 coefficients × 2 stats)
    mfcc_array = np.array(features['mfcc'])
    if len(mfcc_array) > 0:
        mfcc_matrix = mfcc_array.reshape(-1, 13)
        for coef in range(13):
            values = mfcc_matrix[:, coef]
            descriptor.append(float(np.mean(values)))
            descriptor.append(float(np.std(values)))
    else:
        descriptor.extend([0.0] * 26)
    
    # Pitch stats (5 values)
    voiced_pitches = [p for p in features['pitch'] if p > 0]
    if len(voiced_pitches) > 0:
        descriptor.append(float(np.mean(voiced_pitches)))
        descriptor.append(float(np.std(voiced_pitches)))
        descriptor.append(float(np.min(voiced_pitches)))
        descriptor.append(float(np.max(voiced_pitches)))
        descriptor.append(float(len(voiced_pitches) / len(features['pitch'])))
    else:
        descriptor.extend([0.0] * 5)
    
    # Spectral features (6 values)
    descriptor.append(float(np.mean(features['spectral_centroid'])))
    descriptor.append(float(np.std(features['spectral_centroid'])))
    descriptor.append(float(np.mean(features['spectral_rolloff'])))
    descriptor.append(float(np.std(features['spectral_rolloff'])))
    descriptor.append(float(np.mean(features['spectral_flux'])))
    descriptor.append(float(np.std(features['spectral_flux'])))
    
    # Temporal features (4 values)
    descriptor.append(float(np.mean(features['zcr'])))
    descriptor.append(float(np.std(features['zcr'])))
    descriptor.append(float(np.mean(features['energy'])))
    descriptor.append(float(np.std(features['energy'])))
    
    # Chroma features (12 values)
    chroma_array = np.array(features['chroma'])
    if len(chroma_array) > 0:
        chroma_matrix = chroma_array.reshape(-1, 12)
        for note in range(12):
            values = chroma_matrix[:, note]
            descriptor.append(float(np.mean(values)))
    else:
        descriptor.extend([0.0] * 12)
    
    # Delta features (13 values)
    if len(mfcc_array) > 0:
        mfcc_matrix = mfcc_array.reshape(-1, 13)
        mfcc_deltas = calculate_deltas(mfcc_matrix)
        for coef in range(13):
            values = mfcc_deltas[:, coef]
            descriptor.append(float(np.mean(values)))
    else:
        descriptor.extend([0.0] * 13)
    
    # Pad or truncate
    while len(descriptor) < target_dimensions:
        descriptor.append(0.0)
    descriptor = descriptor[:target_dimensions]
    
    # Normalize
    descriptor = normalize_descriptor(descriptor)
    
    return descriptor



def calculate_deltas(features: np.ndarray) -> np.ndarray:
    """Calculate delta features (first derivative)"""
    num_frames = features.shape[0]
    deltas = np.zeros_like(features)
    
    for frame in range(num_frames):
        prev_frame = max(0, frame - 1)
        next_frame = min(num_frames - 1, frame + 1)
        deltas[frame] = (features[next_frame] - features[prev_frame]) / 2
    
    return deltas



def calculate_voice_quality(features: Dict, voicing_ratio: float) -> float:
    """Calculate voice quality score (0-1)"""
    avg_energy = np.mean(features['energy'])
    energy_std = np.std(features['energy'])
    flux_std = np.std(features['spectral_flux'])
    
    voicing_score = voicing_ratio
    energy_score = min(1.0, avg_energy / 0.1)
    stability_score = 1.0 / (1.0 + flux_std)
    
    quality = voicing_score * 0.4 + energy_score * 0.3 + stability_score * 0.3
    return max(0.0, min(1.0, quality))



def normalize_descriptor(descriptor: List[float]) -> List[float]:
    """Normalize descriptor to [0, 1] range"""
    descriptor_array = np.array(descriptor)
    min_val = np.min(descriptor_array)
    max_val = np.max(descriptor_array)
    range_val = max_val - min_val
    
    if range_val == 0:
        return [0.0] * len(descriptor)
    
    normalized = (descriptor_array - min_val) / range_val
    return normalized.tolist()



def create_empty_descriptor(dimensions: int) -> Dict:
    """Create empty descriptor for invalid audio"""
    return {
        'descriptor': [0.0] * dimensions,
        'quality': 0.0,
        'metadata': {
            'avgPitch': 0.0,
            'pitchVariance': 0.0,
            'voicedFrames': 0,
            'totalFrames': 0
        }
    }



def validate_voice_descriptor(descriptor: List[float]) -> bool:
    """Validate voice descriptor"""
    if not isinstance(descriptor, list) or len(descriptor) != 256:
        return False
    
    if any(not isinstance(val, (int, float)) or np.isnan(val) for val in descriptor):
        return False
    
    if not any(abs(val) > 0.001 for val in descriptor):
        return False
    
    mean = np.mean(descriptor)
    variance = np.var(descriptor)
    
    if variance < 0.001:
        return False
    
    return True



def compare_voice_descriptors(desc1: List[float], desc2: List[float]) -> float:
    """Compare two voice descriptors using Euclidean + Cosine distance"""
    if len(desc1) != len(desc2):
        raise ValueError("Descriptors must have same dimensions")
    
    desc1_array = np.array(desc1)
    desc2_array = np.array(desc2)
    
    # Euclidean distance
    euclidean_distance = np.linalg.norm(desc1_array - desc2_array)
    normalized_euclidean = euclidean_distance / np.sqrt(len(desc1))
    
    # Cosine similarity
    dot_product = np.dot(desc1_array, desc2_array)
    mag1 = np.linalg.norm(desc1_array)
    mag2 = np.linalg.norm(desc2_array)
    
    if mag1 == 0 or mag2 == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = dot_product / (mag1 * mag2)
    
    cosine_distance = 1.0 - cosine_sim
    
    # Combined distance (cosine weighted higher - matching TypeScript logic)
    combined_distance = (normalized_euclidean * 0.3) + (cosine_distance * 0.7)
    
    return float(combined_distance)



def validate_speaker_biometrics(desc1: List[float], desc2: List[float]) -> Dict:
    """Validate speaker biometric characteristics (pitch, spectral, timbre)"""
    if len(desc1) != 256 or len(desc2) != 256:
        return {
            'isValid': False,
            'pitchMatch': False,
            'spectralMatch': False,
            'timbreMatch': False,
            'details': 'Invalid dimensions'
        }
    
    # Pitch features (indices 26-30)
    pitch1 = {
        'mean': desc1[26],
        'std': desc1[27],
        'min': desc1[28],
        'max': desc1[29],
        'voicingRatio': desc1[30]
    }
    pitch2 = {
        'mean': desc2[26],
        'std': desc2[27],
        'min': desc2[28],
        'max': desc2[29],
        'voicingRatio': desc2[30]
    }
    
    pitch_mean_diff = abs(pitch1['mean'] - pitch2['mean']) / max(pitch1['mean'], pitch2['mean'], 0.001)
    pitch_range_diff = abs((pitch1['max'] - pitch1['min']) - (pitch2['max'] - pitch2['min'])) / max(pitch1['max'] - pitch1['min'], pitch2['max'] - pitch2['min'], 0.001)
    pitch_match = pitch_mean_diff < 0.10 and pitch_range_diff < 0.15
    
    # Spectral features (indices 31-36)
    spectral1 = {
        'centroidMean': desc1[31],
        'centroidStd': desc1[32],
        'rolloffMean': desc1[33]
    }
    spectral2 = {
        'centroidMean': desc2[31],
        'centroidStd': desc2[32],
        'rolloffMean': desc2[33]
    }
    
    spectral_centroid_diff = abs(spectral1['centroidMean'] - spectral2['centroidMean']) / max(spectral1['centroidMean'], spectral2['centroidMean'], 0.001)
    spectral_rolloff_diff = abs(spectral1['rolloffMean'] - spectral2['rolloffMean']) / max(spectral1['rolloffMean'], spectral2['rolloffMean'], 0.001)
    spectral_match = spectral_centroid_diff < 0.12 and spectral_rolloff_diff < 0.12
    
    # MFCC features (indices 0-25) - Timbre matching
    mfcc1 = np.array(desc1[0:26])
    mfcc2 = np.array(desc2[0:26])
    
    mfcc_dot_product = np.dot(mfcc1, mfcc2)
    mfcc_mag1 = np.linalg.norm(mfcc1)
    mfcc_mag2 = np.linalg.norm(mfcc2)
    
    if mfcc_mag1 == 0 or mfcc_mag2 == 0:
        mfcc_similarity = 0.0
    else:
        mfcc_similarity = mfcc_dot_product / (mfcc_mag1 * mfcc_mag2)
    
    timbre_match = mfcc_similarity > 0.92
    
    is_valid = pitch_match and spectral_match and timbre_match
    details = f"Pitch: {'✓' if pitch_match else '✗'}, Spectral: {'✓' if spectral_match else '✗'}, Timbre: {'✓' if timbre_match else '✗'}"
    
    return {
        'isValid': is_valid,
        'pitchMatch': pitch_match,
        'spectralMatch': spectral_match,
        'timbreMatch': timbre_match,
        'details': details
    }



def base64_to_audio(base64_string: str) -> Tuple[np.ndarray, int]:
    """Convert base64 audio to numpy array"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        audio_bytes = base64.b64decode(base64_string)
        audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        
        return audio_data, sample_rate
        
    except Exception as e:
        raise ValueError(f"Invalid audio format: {str(e)}")



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



def compare_voices(captured_audio_data: np.ndarray, stored_audio_base64: str, sample_rate: int = 16000) -> dict:
    """Compare two voice samples"""
    try:
        print("Extracting descriptor from captured audio...")
        captured_result = extract_voice_descriptor_from_audio(captured_audio_data, sample_rate)
        
        if captured_result['quality'] < 0.50:
            return {
                "matched": False,
                "confidence": 0.0,
                "quality": captured_result['quality'],
                "error": f"Voice quality too low: {round(captured_result['quality'] * 100)}%"
            }
        
        captured_descriptor = captured_result['descriptor']
        
        if not validate_voice_descriptor(captured_descriptor):
            return {
                "matched": False,
                "confidence": 0.0,
                "quality": captured_result['quality'],
                "error": "Invalid voice descriptor"
            }
        
        print("Extracting descriptor from stored audio...")
        stored_audio_data, stored_sr = base64_to_audio(stored_audio_base64)
        stored_result = extract_voice_descriptor_from_audio(stored_audio_data, stored_sr)
        stored_descriptor = stored_result['descriptor']
        
        if not validate_voice_descriptor(stored_descriptor):
            return {
                "matched": False,
                "confidence": 0.0,
                "quality": captured_result['quality'],
                "error": "Invalid stored voice descriptor"
            }
        
        print("Comparing voice descriptors...")
        distance = compare_voice_descriptors(captured_descriptor, stored_descriptor)
        
        biometric_check = validate_speaker_biometrics(captured_descriptor, stored_descriptor)
        
        # Matching thresholds (same as TypeScript)
        MATCH_THRESHOLD = 0.26
        MIN_CONFIDENCE = 78
        
        confidence = max(0, min(100, (1 - distance) * 100))
        
        matched = (distance < MATCH_THRESHOLD and 
                  confidence >= MIN_CONFIDENCE and 
                  biometric_check['isValid'])
        
        print(f"Voice comparison - Distance: {distance:.4f}, Confidence: {confidence:.2f}%, Biometric: {biometric_check['details']}")
        
        return {
            "matched": matched,
            "confidence": round(confidence, 2),
            "quality": round(captured_result['quality'], 2),
            "biometric": biometric_check
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice comparison failed: {str(e)}")



# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VW EduTech Voice Recognition API",
        "version": "1.0",
        "endpoints": {
            "voice-match": "/voice-match (POST)",
            "health": "/health (GET)"
        }
    }



@app.post("/voice-match", response_model=VoiceMatchResponse)
async def voice_match(request: VoiceMatchRequest):
    """Voice matching endpoint"""
    if not request.mob or len(request.mob) < 10:
        raise HTTPException(status_code=400, detail="Invalid mobile number")
    
    if not request.capturedAudio:
        raise HTTPException(status_code=400, detail="Captured audio is required")
    
    try:
        # Step 1: Get student data from external API
        print(f"Fetching student data for mobile: {request.mob}")
        student_data = get_student_data(request.mob)
        
        # Extract stored voice
        stored_voice_base64 = student_data.get("voiceString")
        if not stored_voice_base64:
            raise HTTPException(status_code=404, detail="No voice sample found for this student")
        
        # Step 2: Convert captured audio from base64
        print("Processing captured audio...")
        captured_audio_data, sample_rate = base64_to_audio(request.capturedAudio)
        
        # Step 3: Compare voices
        print("Comparing voices...")
        comparison_result = compare_voices(captured_audio_data, stored_voice_base64, sample_rate)
        
        # Check for errors in comparison
        if "error" in comparison_result:
            raise HTTPException(status_code=400, detail=comparison_result["error"])
        
        # Step 4: Prepare response
        matched = comparison_result["matched"]
        confidence = comparison_result.get("confidence", 0.0)
        quality = comparison_result.get("quality", 0.0)
        
        response = VoiceMatchResponse(
            matched=matched,
            confidence=confidence,
            voiceQuality=quality,
            message="Voice matched successfully!" if matched else "Voice does not match",
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
        "service": "Voice Recognition API"
    }



if __name__ == "__main__":
    import uvicorn
    print("Starting Voice Recognition API...")
    print("Server: http://localhost:8002")
    print("Endpoints:")
    print("  - POST /voice-match - Voice recognition")
    print("  - GET /health - Health check")
    print("API Docs: http://localhost:8002/docs")
    print("Interactive Docs: http://localhost:8002/redoc")
    uvicorn.run(app, host="127.0.0.1", port=8002)