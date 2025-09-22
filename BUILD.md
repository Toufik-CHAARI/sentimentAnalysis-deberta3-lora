# Docker Build Guide

This guide explains how to build the sentiment analysis Docker image locally with AWS credentials.

## Prerequisites

- Docker installed and running
- AWS credentials with access to your S3 bucket
- DVC S3 bucket configured

## Required Environment Variables

You need the following environment variables to build the image:

- `AWS_ACCESS_KEY_ID`: Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key
- `AWS_DEFAULT_REGION`: AWS region (defaults to `eu-west-3`)
- `DVC_BUCKET`: Name of your S3 bucket for DVC storage

## Build Methods

### Method 1: Using the Build Script (Recommended)

The `build-with-secrets.sh` script provides a user-friendly way to build with credentials:

```bash
# Make the script executable (first time only)
chmod +x build-with-secrets.sh

# Build with command line arguments
./build-with-secrets.sh \
  -a "AKIAIOSFODNN7EXAMPLE" \
  -s "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" \
  -b "my-dvc-bucket" \
  -r "eu-west-3"

# Or use long form options
./build-with-secrets.sh \
  --access-key "AKIAIOSFODNN7EXAMPLE" \
  --secret-key "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" \
  --bucket "my-dvc-bucket" \
  --region "eu-west-3"
```

### Method 2: Using Environment Variables

Set the environment variables and use the Makefile:

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="eu-west-3"
export DVC_BUCKET="your-bucket-name"

# Build using Makefile
make build-with-secrets
```

### Method 3: Using a .env File

1. Copy the example environment file:
   ```bash
   cp docs/env.example .env
   ```

2. Edit `.env` with your actual credentials:
   ```bash
   nano .env  # or use your preferred editor
   ```

3. Source the environment file and build:
   ```bash
   source .env
   make build-with-secrets
   ```

### Method 4: Direct Docker Build

```bash
docker build \
  --build-arg AWS_ACCESS_KEY_ID="your-access-key" \
  --build-arg AWS_SECRET_ACCESS_KEY="your-secret-key" \
  --build-arg AWS_DEFAULT_REGION="eu-west-3" \
  --build-arg DVC_BUCKET="your-bucket-name" \
  -t sentiment-analysis-deberta3-lora:latest .
```

## Running the Container

After building, you can run the container:

```bash
# Using Makefile
make run

# Or directly with Docker
docker run -p 8501:8501 sentiment-analysis-deberta3-lora:latest
```

The application will be available at `http://localhost:8501`.

## Troubleshooting

### Build Fails with "UndefinedVar" Warnings

This happens when you use `make build` without providing AWS credentials. Use one of the methods above to provide the required credentials.

### DVC Pull Fails

- Verify your AWS credentials have access to the S3 bucket
- Check that the bucket name is correct
- Ensure the bucket exists and contains the model files
- Verify your AWS region is correct

### Permission Denied on Script

If you get permission denied when running the build script:

```bash
chmod +x build-with-secrets.sh
```

## Security Notes

- Never commit your `.env` file or hardcode credentials in scripts
- Use IAM roles with minimal required permissions
- Consider using AWS Secrets Manager for production deployments
- The build script masks the access key in output for security

## Available Makefile Targets

- `make build`: Build without credentials (will fail at DVC pull)
- `make build-with-secrets`: Build with AWS credentials from environment
- `make run`: Run the container locally
- `make run-detached`: Run container in background
- `make stop`: Stop running container
- `make logs`: Show container logs
- `make help`: Show all available commands
