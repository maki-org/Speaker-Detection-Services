# api_server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import tempfile
import shutil
from datetime import datetime
import numpy as np
from collections import defaultdict
import json
from pydub import AudioSegment
from speechbrain.inference.speaker import EncoderClassifier
from groq import Groq
from sklearn.cluster import AgglomerativeClustering
import uvicorn
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MAKI.AI Speaker Diarization API",
    description="Multi-speaker identification and diarization service",
    version="1.0.0"
)

# Global variables for models (loaded once at startup)
groq_client = None
classifier = None

# Response models
class ProcessingStatus(BaseModel):
    status: str
    message: str
    job_id: str = None
    result_url: str = None

class SpeakerInfo(BaseModel):
    segment_count: int
    total_time_seconds: float
    percentage: float
    is_user: bool

class SegmentInfo(BaseModel):
    segment_id: int
    start_time: float
    end_time: float
    duration: float
    speaker: str
    similarity_to_user: float
    transcript: str

class DiarizationResult(BaseModel):
    conversation_id: str
    processing_date: str
    audio_duration_seconds: float
    total_speakers: int
    total_segments: int
    speakers: dict[str, SpeakerInfo]
    segments: list[SegmentInfo]


@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup"""
    global groq_client, classifier
    
    logger.info("🚀 Starting MAKI.AI Diarization API...")
    
    # Initialize Groq
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")
    
    groq_client = Groq(api_key=groq_api_key)
    logger.info("✅ Groq client initialized")
    
    # Load SpeechBrain model
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}  # Change to "cuda" if GPU available
    )
    logger.info("✅ SpeechBrain speaker recognition loaded")
    
    # Create output directory
    Path("./processing_output").mkdir(exist_ok=True)
    logger.info("✅ Output directory ready")


def extract_user_embedding(enrollment_audio_path: str) -> np.ndarray:
    """Extract speaker embedding from enrollment audio"""
    try:
        # Convert to proper WAV format first using pydub
        audio = AudioSegment.from_file(enrollment_audio_path)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio.set_channels(1).set_frame_rate(16000).export(tmp.name, format="wav")
            tmp_path = tmp.name
        
        # Load with SpeechBrain
        user_signal = classifier.load_audio(tmp_path)
        user_embedding = classifier.encode_batch(user_signal).squeeze().detach().cpu().numpy()
        
        # Clean up
        os.remove(tmp_path)
        
        return user_embedding
    except Exception as e:
        logger.error(f"Error extracting embedding: {str(e)}")
        raise Exception(f"Error loading audio file: {str(e)}")


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using Groq Whisper"""
    with open(audio_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language="en",
            temperature=0.0
        )
    return transcription


def relabel_speakers_by_order(segment_data: list) -> list:
    """Relabel speakers in order of first appearance"""
    speaker_first_appearance = {}
    speaker_mapping = {}
    next_speaker_num = 1
    
    for seg in segment_data:
        current_speaker = seg['speaker']
        
        # Skip USER
        if current_speaker == 'USER':
            continue
            
        # Record first appearance
        if current_speaker not in speaker_first_appearance:
            speaker_first_appearance[current_speaker] = seg['start']
    
    # Sort speakers by first appearance
    sorted_speakers = sorted(speaker_first_appearance.items(), key=lambda x: x[1])
    
    # Create mapping
    for old_label, _ in sorted_speakers:
        speaker_mapping[old_label] = f'Speaker {next_speaker_num}'
        next_speaker_num += 1
    
    # Apply mapping
    for seg in segment_data:
        if seg['speaker'] != 'USER':
            seg['speaker'] = speaker_mapping.get(seg['speaker'], seg['speaker'])
    
    return segment_data

def process_diarization(
    conversation_audio: AudioSegment,
    user_embedding: np.ndarray,
    segments: list,
    output_folder: str
) -> tuple:
    """Process speaker diarization on conversation audio"""
    
    segment_data = []
    
    # Extract embeddings for all segments
    for idx, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        text = seg['text'].strip()
        
        seg_audio = conversation_audio[start * 1000 : end * 1000]
        
        if len(seg_audio) < 500:
            continue
        
        # Extract segment embedding
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg_audio.set_channels(1).set_frame_rate(16000).export(tmp.name, format="wav")
            seg_signal = classifier.load_audio(tmp.name)
            seg_embedding = classifier.encode_batch(seg_signal).squeeze().detach().cpu().numpy()
            os.remove(tmp.name)
        
        # Calculate similarity to USER
        similarity = 1 - np.dot(user_embedding, seg_embedding) / (
            np.linalg.norm(user_embedding) * np.linalg.norm(seg_embedding)
        )
        
        segment_data.append({
            'start': start,
            'end': end,
            'text': text,
            'embedding': seg_embedding,
            'similarity': float(similarity),
            'audio': seg_audio
        })
    
    # Smart USER classification
    similarities = [seg['similarity'] for seg in segment_data]
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    user_threshold = min(mean_sim + 0.5 * std_sim, 0.75)
    
    user_segments = []
    other_segments = []
    
    for seg in segment_data:
        if seg['similarity'] < user_threshold:
            seg['speaker'] = 'USER'
            user_segments.append(seg)
        else:
            other_segments.append(seg)
    
    # Cluster OTHER speakers
    if len(other_segments) > 1:
        other_embeddings = np.array([seg['embedding'] for seg in other_segments])
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.70,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = list(clustering.fit_predict(other_embeddings))
        
        # Smart merging of small clusters
        cluster_counts = {}
        for label in cluster_labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        
        small_clusters = [c for c, count in cluster_counts.items() if count <= 2]
        large_clusters = [c for c, count in cluster_counts.items() if count > 2]
        
        if small_clusters and large_clusters:
            for small_c in small_clusters:
                small_indices = [i for i, label in enumerate(cluster_labels) if label == small_c]
                small_embs = other_embeddings[small_indices]
                
                best_match = None
                best_distance = float('inf')
                
                for large_c in large_clusters:
                    large_indices = [i for i, label in enumerate(cluster_labels) if label == large_c]
                    large_embs = other_embeddings[large_indices]
                    
                    distances = []
                    for small_emb in small_embs:
                        for large_emb in large_embs:
                            dist = 1 - np.dot(small_emb, large_emb) / (
                                np.linalg.norm(small_emb) * np.linalg.norm(large_emb)
                            )
                            distances.append(dist)
                    avg_dist = np.mean(distances)
                    
                    if avg_dist < best_distance:
                        best_distance = avg_dist
                        best_match = large_c
                
                for i in small_indices:
                    cluster_labels[i] = best_match
        
        # Renumber clusters
        unique_clusters = sorted(set(cluster_labels))
        cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
        cluster_labels = [cluster_mapping[label] for label in cluster_labels]
        
        # Assign speaker labels
        for seg, cluster_id in zip(other_segments, cluster_labels):
            seg['speaker'] = f'Speaker {cluster_id + 1}'
    
    elif len(other_segments) == 1:
        other_segments[0]['speaker'] = 'Speaker 1'
    
    

    # Combine and sort
    all_segments = user_segments + other_segments

    # Relabel speakers by order of appearance
    all_segments = relabel_speakers_by_order(all_segments)

    all_segments.sort(key=lambda x: x['start'])
    
    # Save segments
    segment_files = []
    matches = []
    
    for idx, seg in enumerate(all_segments):
        segment_filename = f"segment_{idx+1:03d}_{seg['speaker'].replace(' ', '_')}_{seg['start']:.2f}-{seg['end']:.2f}.wav"
        segment_path = os.path.join(output_folder, segment_filename)
        seg['audio'].export(segment_path, format="wav")
        
        matches.append((seg['start'], seg['end'], seg['similarity'], seg['speaker'], seg['text']))
        segment_files.append(segment_path)
    
    return matches, segment_files


@app.post("/diarize", response_model=DiarizationResult)
async def diarize_conversation(
    enrollment_audio: UploadFile = File(..., description="User's voice sample for enrollment"),
    conversation_audio: UploadFile = File(..., description="Conversation audio to analyze")
):
    """
    Process speaker diarization on a conversation
    
    - **enrollment_audio**: Audio file containing the user's voice (for identification)
    - **conversation_audio**: Audio file containing the multi-speaker conversation
    
    Returns detailed diarization results with speaker identification
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"job_{timestamp}"
    output_folder = f"./processing_output/{job_id}"
    
    try:
        # Create job folder
        os.makedirs(output_folder, exist_ok=True)
        
        logger.info(f"📋 Processing job: {job_id}")
        
        # Save uploaded files with original extensions
        enrollment_ext = os.path.splitext(enrollment_audio.filename)[1] or '.mp3'
        conversation_ext = os.path.splitext(conversation_audio.filename)[1] or '.mp3'
        
        enrollment_path = os.path.join(output_folder, f"enrollment{enrollment_ext}")
        conversation_path = os.path.join(output_folder, f"conversation{conversation_ext}")
        
        with open(enrollment_path, "wb") as f:
            f.write(await enrollment_audio.read())
        
        with open(conversation_path, "wb") as f:
            f.write(await conversation_audio.read())
        
        logger.info("✅ Files uploaded")
        
        # Extract user embedding
        logger.info("🎤 Extracting user voice embedding...")
        user_embedding = extract_user_embedding(enrollment_path)
        
        # Load and prepare conversation audio
        logger.info("📄 Loading conversation audio...")
        audio = AudioSegment.from_file(conversation_path)
        duration_seconds = len(audio) / 1000
        
        # Convert to WAV for transcription
        wav_path = os.path.join(output_folder, "temp_conversation.wav")
        audio.set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")
        
        # Transcribe
        logger.info("🔍 Transcribing with Groq Whisper...")
        transcription = transcribe_audio(wav_path)
        segments = transcription.segments
        
        logger.info(f"✅ Transcription complete: {len(segments)} segments")
        
        # Process diarization
        logger.info("🔍 Analyzing speakers...")
        matches, segment_files = process_diarization(
            audio, user_embedding, segments, output_folder
        )
        
        # Calculate statistics
        speaker_counts = defaultdict(int)
        speaker_times = defaultdict(float)
        
        for start, end, sim, speaker, text in matches:
            speaker_counts[speaker] += 1
            speaker_times[speaker] += (end - start)
        
        total_time = sum(speaker_times.values())
        
        # Build response
        result = DiarizationResult(
            conversation_id=job_id,
            processing_date=datetime.now().isoformat(),
            audio_duration_seconds=float(duration_seconds),
            total_speakers=len(speaker_counts),
            total_segments=len(matches),
            speakers={
                speaker: SpeakerInfo(
                    segment_count=int(speaker_counts[speaker]),
                    total_time_seconds=float(speaker_times[speaker]),
                    percentage=float((speaker_times[speaker] / total_time * 100) if total_time > 0 else 0),
                    is_user=speaker == "USER"
                )
                for speaker in sorted(speaker_counts.keys())
            },
            segments=[
                SegmentInfo(
                    segment_id=idx + 1,
                    start_time=float(start),
                    end_time=float(end),
                    duration=float(end - start),
                    speaker=speaker,
                    similarity_to_user=float(sim),
                    transcript=text
                )
                for idx, (start, end, sim, speaker, text) in enumerate(matches)
            ]
        )
        
        # Save JSON result
        json_path = os.path.join(output_folder, "result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Job {job_id} completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error processing job {job_id}: {str(e)}")
        # Cleanup on error
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download all processing results as a ZIP file"""
    
    output_folder = f"./processing_output/{job_id}"
    
    if not os.path.exists(output_folder):
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Create ZIP
    zip_path = f"{output_folder}.zip"
    shutil.make_archive(output_folder, 'zip', output_folder)
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{job_id}_results.zip"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "groq_client": groq_client is not None,
        "classifier": classifier is not None
    }


@app.get("/")
async def root():
    """API information"""
    return {
        "service": "MAKI.AI Speaker Diarization API",
        "version": "1.0.0",
        "endpoints": {
            "POST /diarize": "Process speaker diarization",
            "GET /download/{job_id}": "Download results",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=7000,
        reload=False,
        workers=1
    )