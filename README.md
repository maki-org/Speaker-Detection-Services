# Speaker-Detection-Services

₹₹
docker build -t maki-diarization . && docker run -it \
 --name maki-api \
 -p 8000:7000 \
 -e GROQ_API_KEY="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" \
 maki-diarization
