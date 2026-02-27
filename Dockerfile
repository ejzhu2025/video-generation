FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg for video+audio merging
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .
COPY .env.example .

# Create uploads directory (ephemeral, images are pushed to fal CDN on I2V)
RUN mkdir -p uploads

# HuggingFace Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
