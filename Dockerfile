FROM python:3.11-slim

# Metadata
LABEL maintainer="devops-openenv"
LABEL description="DevOps Incident Response OpenEnv – Hugging Face Space"

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Ensure Python can find modules at the repo root (env.py, tasks.py, etc.)
ENV PYTHONPATH=/app

# Expose the port used by Hugging Face Spaces
EXPOSE 7860

# Start the FastAPI server via uvicorn
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
