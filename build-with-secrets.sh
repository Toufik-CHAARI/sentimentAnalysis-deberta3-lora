#!/bin/bash

# Build script for sentiment analysis Docker image with AWS credentials
# Usage: ./build-with-secrets.sh [OPTIONS]
#
# Options:
#   -a, --access-key     AWS Access Key ID
#   -s, --secret-key     AWS Secret Access Key
#   -r, --region         AWS Region (default: eu-west-3)
#   -b, --bucket         DVC S3 Bucket name
#   -t, --tag            Docker image tag (default: latest)
#   -h, --help           Show this help message

set -e

# Default values
AWS_REGION="eu-west-3"
IMAGE_TAG="latest"
IMAGE_NAME="sentiment-analysis-deberta3-lora"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
Build script for sentiment analysis Docker image with AWS credentials

Usage: $0 [OPTIONS]

Options:
  -a, --access-key     AWS Access Key ID (required)
  -s, --secret-key     AWS Secret Access Key (required)
  -r, --region         AWS Region (default: eu-west-3)
  -b, --bucket         DVC S3 Bucket name (required)
  -t, --tag            Docker image tag (default: latest)
  -h, --help           Show this help message

Examples:
  $0 -a AKIAIOSFODNN7EXAMPLE -s wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY -b my-dvc-bucket
  $0 --access-key AKIAIOSFODNN7EXAMPLE --secret-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY --bucket my-dvc-bucket --region us-east-1

Environment Variables:
  You can also set these environment variables instead of using command line options:
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - AWS_DEFAULT_REGION
  - DVC_S3_BUCKET

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--access-key)
            AWS_ACCESS_KEY_ID="$2"
            shift 2
            ;;
        -s|--secret-key)
            AWS_SECRET_ACCESS_KEY="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -b|--bucket)
            DVC_S3_BUCKET="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if required variables are set
if [[ -z "$AWS_ACCESS_KEY_ID" ]]; then
    print_error "AWS Access Key ID is required. Use -a or --access-key option or set AWS_ACCESS_KEY_ID environment variable."
    exit 1
fi

if [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    print_error "AWS Secret Access Key is required. Use -s or --secret-key option or set AWS_SECRET_ACCESS_KEY environment variable."
    exit 1
fi

if [[ -z "$DVC_S3_BUCKET" ]]; then
    print_error "DVC S3 Bucket name is required. Use -b or --bucket option or set DVC_S3_BUCKET environment variable."
    exit 1
fi

# Set AWS_DEFAULT_REGION
AWS_DEFAULT_REGION="$AWS_REGION"

print_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
print_info "AWS Region: ${AWS_DEFAULT_REGION}"
print_info "DVC Bucket: ${DVC_S3_BUCKET}"
print_info "AWS Access Key ID: ${AWS_ACCESS_KEY_ID:0:8}..."

# Build the Docker image with build arguments
docker build \
    --build-arg AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    --build-arg AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --build-arg AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
    --build-arg DVC_S3_BUCKET="$DVC_S3_BUCKET" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

if [[ $? -eq 0 ]]; then
    print_success "Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
    print_info "You can now run the container with:"
    echo "  docker run -p 8501:8501 ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    print_info "Or use the Makefile:"
    echo "  make run"
    echo "  make run-detached"
else
    print_error "Docker build failed!"
    exit 1
fi
