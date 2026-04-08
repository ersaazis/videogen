# Use a slim Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# - ffmpeg: for moviepy
# - libsndfile1: for librosa/DeepFilterNet
# - fonts-dejavu: for PIL subtitle generation
# - libgl1 & libglib2.0-0: for OpenCV/PIL dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    fonts-dejavu-core \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
