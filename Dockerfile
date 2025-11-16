FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including sox for torchaudio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    sox \
    libsox-dev \
    libsox-fmt-all \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy api_server.py
COPY api_server.py .

# Create output directory
RUN mkdir -p ./processing_output

# Expose port
EXPOSE 7000

# Run the application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7000"]