# api_server.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
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
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MAKI.AI Speaker Diarization API (Enroll + Diarize)",
    description="Multi-speaker identification and diarization service with separate enroll endpoint",
    version="1.1.0"
)

# Global variables for models (loaded once at startup)
groq_client = None
classifier = None

# Directories
OUTPUT_DIR = Path("./processing_output")
ENROLL_DIR = Path("./enrollments")
OUTPUT_DIR.mkdir(exist_ok=True)
ENROLL_DIR.mkdir(exist_ok=True)

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

    logger.info("🚀 Starting MAKI.AI Diarization API (Enroll + Diarize)...")

    # Initialize Groq
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        raise RuntimeError("GROQ_API_KEY environment variable not set")

    groq_client = Groq(api_key=groq_api_key)
    logger.info("✅ Groq client initialized")

    # Load SpeechBrain model (encoder)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}  # change to "cuda" if GPU available
    )
    logger.info("✅ SpeechBrain speaker recognition loaded")

    # Create needed directories (already created above)
    logger.info("✅ Output and enrollment directories ready")


def extract_user_embedding(enrollment_audio_path: str) -> np.ndarray:
    """Extract speaker embedding from enrollment audio file path"""
    try:
        # Convert input audio to mono WAV 16k
        audio = AudioSegment.from_file(enrollment_audio_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio.set_channels(1).set_frame_rate(16000).export(tmp.name, format="wav")
            tmp_path = tmp.name

        # Use speechbrain classifier to load and encode
        user_signal = classifier.load_audio(tmp_path)
        user_embedding = classifier.encode_batch(user_signal).squeeze().detach().cpu().numpy()

        # Clean up
        os.remove(tmp_path)
        return user_embedding
    except Exception as e:
        logger.exception("Failed to extract embedding")
        raise Exception(f"Error extracting embedding: {str(e)}")


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using Groq Whisper. Returns transcription object (expected to have .segments)."""
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
    """Relabel speakers in order of first appearance (skip USER as special)"""
    speaker_first_appearance = {}
    for seg in segment_data:
        current_speaker = seg.get('speaker')
        if current_speaker == 'USER':
            continue
        if current_speaker not in speaker_first_appearance:
            speaker_first_appearance[current_speaker] = seg['start']
    sorted_speakers = sorted(speaker_first_appearance.items(), key=lambda x: x[1])
    speaker_mapping = {}
    next_num = 1
    for old_label, _ in sorted_speakers:
        speaker_mapping[old_label] = f"Speaker {next_num}"
        next_num += 1
    for seg in segment_data:
        if seg.get('speaker') != 'USER' and seg.get('speaker') in speaker_mapping:
            seg['speaker'] = speaker_mapping[seg['speaker']]
    return segment_data


def process_diarization(
    conversation_audio: AudioSegment,
    user_embedding: np.ndarray,
    segments: list,
    output_folder: str
) -> tuple:
    """Core diarization processing (same logic as provided), returns matches (list) and segment file paths"""
    segment_data = []

    # Extract embeddings for all transcription segments
    for idx, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        text = seg.get('text', '').strip()

        seg_audio = conversation_audio[int(start * 1000) : int(end * 1000)]

        if len(seg_audio) < 500:  # skip tiny segments (<0.5s)
            continue

        # Create temp wav and extract embedding
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            seg_audio.set_channels(1).set_frame_rate(16000).export(tmp.name, format="wav")
            seg_signal = classifier.load_audio(tmp.name)
            seg_embedding = classifier.encode_batch(seg_signal).squeeze().detach().cpu().numpy()
            os.remove(tmp.name)

        # cosine distance (1 - cosine similarity)
        similarity = 1 - np.dot(user_embedding, seg_embedding) / (
            np.linalg.norm(user_embedding) * np.linalg.norm(seg_embedding) + 1e-12
        )

        segment_data.append({
            'start': float(start),
            'end': float(end),
            'text': text,
            'embedding': seg_embedding,
            'similarity': float(similarity),
            'audio': seg_audio
        })

    if not segment_data:
        return [], []

    # Decide USER vs OTHERS
    similarities = [s['similarity'] for s in segment_data]
    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    user_threshold = min(mean_sim + 0.5 * std_sim, 0.75)  # heuristic

    user_segments = []
    other_segments = []
    for seg in segment_data:
        if seg['similarity'] < user_threshold:
            seg['speaker'] = 'USER'
            user_segments.append(seg)
        else:
            other_segments.append(seg)

    # Cluster other speakers using AgglomerativeClustering (cosine)
    if len(other_segments) > 1:
        other_embeddings = np.array([s['embedding'] for s in other_segments])
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.70,
            metric='cosine',
            linkage='average'
        )
        cluster_labels = list(clustering.fit_predict(other_embeddings))

        # Merge small clusters into best large cluster (heuristic from your original code)
        cluster_counts = {}
        for lab in cluster_labels:
            cluster_counts[lab] = cluster_counts.get(lab, 0) + 1
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
                    for se in small_embs:
                        for le in large_embs:
                            dist = 1 - np.dot(se, le) / (np.linalg.norm(se) * np.linalg.norm(le) + 1e-12)
                            distances.append(dist)
                    avg_dist = np.mean(distances)
                    if avg_dist < best_distance:
                        best_distance = avg_dist
                        best_match = large_c
                for i in small_indices:
                    cluster_labels[i] = best_match

        # Renumber clusters 0..N-1
        unique = sorted(set(cluster_labels))
        mapping = {old: new for new, old in enumerate(unique)}
        cluster_labels = [mapping[l] for l in cluster_labels]

        for seg, lbl in zip(other_segments, cluster_labels):
            seg['speaker'] = f"Speaker {lbl + 1}"
    elif len(other_segments) == 1:
        other_segments[0]['speaker'] = 'Speaker 1'

    all_segments = user_segments + other_segments
    all_segments = relabel_speakers_by_order(all_segments)
    all_segments.sort(key=lambda x: x['start'])

    segment_files = []
    matches = []
    for idx, seg in enumerate(all_segments):
        speaker_safe = seg['speaker'].replace(" ", "_")
        segment_filename = f"segment_{idx+1:03d}_{speaker_safe}_{seg['start']:.2f}-{seg['end']:.2f}.wav"
        segment_path = os.path.join(output_folder, segment_filename)
        seg['audio'].export(segment_path, format="wav")
        matches.append((seg['start'], seg['end'], seg['similarity'], seg['speaker'], seg['text']))
        segment_files.append(segment_path)

    return matches, segment_files


@app.post("/enroll")
async def enroll_user(enrollment_audio: UploadFile = File(..., description="User's voice sample for enrollment")):
    """
    Enroll a user voice sample and return the extracted embedding + enrollment_id.

    - **enrollment_audio**: audio file containing user's voice (wav/mp3/...)
    Returns:
      {
        "enrollment_id": "<id>",
        "embedding": [ ... list of floats ... ],
        "message": "enrollment saved"
      }
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    enroll_id = f"enroll_{timestamp}"
    enroll_folder = ENROLL_DIR / enroll_id
    enroll_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded file
        ext = os.path.splitext(enrollment_audio.filename)[1] or ".mp3"
        enroll_path = str(enroll_folder / f"enrollment{ext}")
        with open(enroll_path, "wb") as f:
            f.write(await enrollment_audio.read())

        logger.info(f"Saved enrollment audio to {enroll_path}")

        # Extract embedding
        embedding = extract_user_embedding(enroll_path)  # returns numpy array

        # Save numpy file for future use
        npy_path = enroll_folder / "embedding.npy"
        np.save(str(npy_path), embedding)
        logger.info(f"Saved enrollment embedding to {npy_path}")

        # Return embedding as list (JSON serializable)
        return {
            "enrollment_id": enroll_id,
            "embedding": embedding.tolist(),
            "message": "enrollment saved"
        }
    except Exception as e:
        logger.exception("Enrollment failed")
        # cleanup
        if enroll_folder.exists():
            shutil.rmtree(enroll_folder)
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@app.post("/diarize", response_model=DiarizationResult)
async def diarize_conversation(
    conversation_audio: UploadFile = File(..., description="Conversation audio to analyze"),
    enrollment_id: Optional[str] = Form(None, description="Enrollment ID returned by /enroll (optional)"),
    user_embedding: Optional[str] = Form(None, description="JSON string of the user embedding array (optional)")
):
    """
    Diarize a conversation using a supplied user embedding.
    Provide either:
      - enrollment_id (string) referencing a prior /enroll result, OR
      - user_embedding (JSON string of embedding list)
    Plus:
      - conversation_audio (file)

    Returns the same detailed diarization JSON response as before.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"job_{timestamp}"
    output_folder = OUTPUT_DIR / job_id
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Decide which embedding to use
        embedding_np = None
        if enrollment_id:
            enroll_path = ENROLL_DIR / enrollment_id / "embedding.npy"
            if not enroll_path.exists():
                raise HTTPException(status_code=404, detail="enrollment_id not found")
            embedding_np = np.load(str(enroll_path))
            logger.info(f"Loaded embedding from enrollment {enrollment_id}")
        elif user_embedding:
            try:
                emb_list = json.loads(user_embedding)
                embedding_np = np.array(emb_list, dtype=np.float32)
                logger.info("Loaded embedding from provided form field")
            except Exception as e:
                logger.exception("Failed to parse user_embedding form field")
                raise HTTPException(status_code=400, detail="user_embedding must be a valid JSON array string")
        else:
            raise HTTPException(status_code=400, detail="Provide either enrollment_id or user_embedding")

        # Save conversation audio
        conv_ext = os.path.splitext(conversation_audio.filename)[1] or ".mp3"
        conversation_path = str(output_folder / f"conversation{conv_ext}")
        with open(conversation_path, "wb") as f:
            f.write(await conversation_audio.read())
        logger.info(f"Saved conversation audio to {conversation_path}")

        # Load conversation audio via pydub
        audio = AudioSegment.from_file(conversation_path)
        duration_seconds = len(audio) / 1000.0

        # Convert to WAV 16k for transcription
        wav_path = str(output_folder / "temp_conversation.wav")
        audio.set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")

        # Transcribe with Groq Whisper
        logger.info("🔍 Transcribing conversation with Groq Whisper...")
        transcription = transcribe_audio(wav_path)

        # transcription expected to have .segments (list of dicts with start, end, text)
        if not hasattr(transcription, "segments") and not isinstance(transcription, dict):
            logger.warning("transcription response structure unexpected; attempting to treat as dict")
        segments = getattr(transcription, "segments", None) or transcription.get("segments", None)
        if not segments:
            raise Exception("No transcription segments returned")

        logger.info(f"✅ Transcription produced {len(segments)} segments")

        # Process diarization using provided user embedding
        logger.info("🔍 Running speaker analysis...")
        matches, segment_files = process_diarization(audio, embedding_np, segments, str(output_folder))

        # Calculate statistics
        speaker_counts = defaultdict(int)
        speaker_times = defaultdict(float)
        for start, end, sim, speaker, text in matches:
            speaker_counts[speaker] += 1
            speaker_times[speaker] += (end - start)
        total_time = sum(speaker_times.values()) or 1.0

        # Build response object
        speakers_dict = {
            speaker: SpeakerInfo(
                segment_count=int(speaker_counts[speaker]),
                total_time_seconds=float(speaker_times[speaker]),
                percentage=float((speaker_times[speaker] / total_time * 100)),
                is_user=(speaker == "USER")
            )
            for speaker in sorted(speaker_counts.keys())
        }

        segments_list = [
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

        result = DiarizationResult(
            conversation_id=job_id,
            processing_date=datetime.now().isoformat(),
            audio_duration_seconds=float(duration_seconds),
            total_speakers=len(speaker_counts),
            total_segments=len(matches),
            speakers=speakers_dict,
            segments=segments_list
        )

        # Save JSON to disk for download
        json_path = output_folder / "result.json"
        with open(str(json_path), "w", encoding="utf-8") as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Job {job_id} completed successfully")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Processing failed for job {job_id}")
        # cleanup on error
        if output_folder.exists():
            shutil.rmtree(output_folder)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download all processing results as a ZIP file"""
    output_folder = OUTPUT_DIR / job_id
    if not output_folder.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    zip_path = str(OUTPUT_DIR / f"{job_id}.zip")
    shutil.make_archive(str(output_folder), 'zip', str(output_folder))
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
        "version": "1.1.0",
        "endpoints": {
            "POST /enroll": "Upload enrollment audio -> returns enrollment_id + embedding",
            "POST /diarize": "Diarize using enrollment_id OR user_embedding + conversation_audio",
            "GET /download/{job_id}": "Download results zip",
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
