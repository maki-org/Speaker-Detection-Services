# Speaker-Detection-Services

#### to run

```bash
docker build -t maki-diarization . && docker run -it \
 --name maki-api \
 -p 8000:7000 \
 -e GROQ_API_KEY="-groq-api-key" \
 maki-diarization
```

### after open the index.html and test
