# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC with S3 support
RUN pip install --no-cache-dir dvc[s3]

# Copy DVC files (excluding config for security)
COPY .dvcignore .dvcignore
COPY deberta3_lora_400000k.dvc deberta3_lora_400000k.dvc

# Accept build arguments for AWS credentials and DVC configuration
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION=eu-west-3
ARG DVC_BUCKET=sentiment-analysis-deberta3-lora

# Set environment variables for DVC
ENV DVC_BUCKET=${DVC_BUCKET}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
ENV DOCKER_CONTAINER=true

# Initialize Git repository, configure DVC remote, and pull the model from S3
RUN git init && \
    git config user.email "docker@example.com" && \
    git config user.name "Docker User" && \
    dvc remote add -d storage s3://${DVC_BUCKET} && \
    dvc remote modify storage access_key_id ${AWS_ACCESS_KEY_ID} && \
    dvc remote modify storage secret_access_key ${AWS_SECRET_ACCESS_KEY} && \
    dvc remote modify storage region ${AWS_DEFAULT_REGION} && \
    dvc pull deberta3_lora_400000k.dvc

# Copy the rest of the application
COPY app_refactored.py .
COPY src/ src/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app
CMD ["streamlit", "run", "app_refactored.py", "--server.port=8501", "--server.address=0.0.0.0"]
