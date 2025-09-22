# Sentiment Analysis with DeBERTa v3 + LoRA

A production-ready Streamlit application for sentiment analysis using a fine-tuned DeBERTa v3 model with LoRA (Low-Rank Adaptation). The model is stored in S3 and pulled via DVC during Docker image building. Features automated CI/CD deployment to AWS EC2 with ECR.

![Demo](deberta3lorav2.gif)
![Demo](deberta3-lora-stat.gif)


## 🚀 Features

- **Advanced Sentiment Analysis**: Uses DeBERTa v3 with LoRA fine-tuning for high accuracy
- **Interactive Web Interface**: Built with Streamlit for easy text input and visualization
- **Model Management**: DVC integration for versioned model storage in S3
- **Docker Support**: Containerized application for consistent deployment
- **AWS Integration**: Ready for deployment on AWS EC2 with ECR
- **CI/CD Pipeline**: Automated deployment with GitHub Actions

## 📋 Prerequisites

- Python 3.12+
- Docker
- AWS CLI (for deployment)
- DVC (for model management)

## 🛠️ Local Development

### 1. Clone the Repository

```bash
git clone <repository-url>
cd sentimentAnalysis-deberta3-lora
```

### 2. Set Up Environment Variables

Create a `.env` file with your AWS credentials:

```bash
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
DVC_BUCKET=your-s3-bucket-name
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Model

```bash
dvc pull deberta3_lora_400000k.dvc
```

### 5. Run the Application

```bash
streamlit run app_refactored.py
```

The application will be available at `http://localhost:8501`.

## 🐳 Docker Deployment

### Local Docker Build

```bash
# Build the Docker image with AWS credentials
make build-with-secrets

# Or use the build script
./build-with-secrets.sh -a "your-access-key" -s "your-secret-key" -b "your-bucket"

# Run the container locally
make run

# Or run in detached mode
make run-detached
```

### Available Make Commands

```bash
make help                    # Show all available commands
make build                   # Build Docker image locally (without AWS credentials)
make build-with-secrets      # Build Docker image with AWS credentials
make run                     # Run Docker container locally
make run-detached           # Run Docker container in background
make stop                    # Stop running container
make logs                    # Show container logs
make shell                   # Open shell in running container
make clean                   # Clean up Docker resources
make test-local              # Build and test locally
make test                    # Run tests with pytest
make test-coverage           # Run tests with coverage report
make lint                    # Run linting with flake8
make format                  # Format code with black and isort
make type-check              # Run type checking with mypy
```

## ☁️ AWS Deployment

### 1. Set Up AWS ECR

```bash
# Login to AWS ECR
make ecr-login

# Create ECR repository
make ecr-create-repo

# Tag and push image to ECR
make push-ec2
```

### 2. Deploy to EC2

The GitHub Actions workflow will automatically deploy to EC2 when you push to the main branch.

### Required GitHub Secrets

Set up the following secrets in your GitHub repository:

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `DVC_S3_BUCKET`: Your S3 bucket name for DVC
- `EC2_INSTANCE_ID`: Your EC2 instance ID (e.g., i-0e8847be115811a7c)

### Manual EC2 Deployment

If you prefer manual deployment:

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker (if not already installed)
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu

# Install AWS CLI (if not already installed)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure

# Get your account ID and login to ECR
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region eu-west-3 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.eu-west-3.amazonaws.com

# Stop and remove existing container (if any)
sudo docker stop sentiment-analysis-app || true
sudo docker rm sentiment-analysis-app || true

# Clean up old images
sudo docker image prune -af

# Pull and run the latest image
docker pull $ACCOUNT_ID.dkr.ecr.eu-west-3.amazonaws.com/sentiment-analysis-deberta3-lora:latest
docker run -d --name sentiment-analysis-app --restart unless-stopped -p 80:8501 \
  -e AWS_ACCESS_KEY_ID=your-access-key \
  -e AWS_SECRET_ACCESS_KEY=your-secret-key \
  -e AWS_DEFAULT_REGION=eu-west-3 \
  -e DVC_BUCKET=your-bucket \
  $ACCOUNT_ID.dkr.ecr.eu-west-3.amazonaws.com/sentiment-analysis-deberta3-lora:latest
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key for S3/DVC | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for S3/DVC | Required |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `DVC_BUCKET` | S3 bucket name for DVC | Required |

### Model Configuration

The model is configured in `src/config.py`:

- **Model Path**: `deberta3_lora_400000k/merged/`
- **Labels**:
  - 0: "😞 Negative Sentiment"
  - 1: "😊 Positive Sentiment"
- **Batch Size**: 32
- **Max Length**: 128

## 📊 Usage

1. **Text Input**: Enter text in the text area
2. **Analysis**: Click "Analyze Sentiment" to get predictions
3. **Visualization**: View sentiment distribution and confidence scores
4. **Batch Processing**: Upload CSV files for bulk analysis
5. **EDA**: Explore the dataset with interactive visualizations

## 🏗️ Project Structure

```
sentimentAnalysis-deberta3-lora/
├── src/                          # Modular application code
│   ├── config.py                 # Configuration settings
│   ├── data_service.py           # Data handling and preprocessing
│   ├── model_service.py          # Model loading and inference
│   ├── utils.py                  # Utility functions
│   └── visualization_service.py  # Plotting and visualization
├── tests/                        # Test files
├── docs/                         # Documentation
├── .github/workflows/            # CI/CD pipelines
├── app_refactored.py             # Main Streamlit application
├── Dockerfile                    # Docker configuration
├── Makefile                      # Build and deployment commands
├── build-with-secrets.sh         # Secure build script
└── requirements.txt              # Python dependencies
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Docker        │    │   AWS S3        │
│   Frontend      │◄──►│   Container     │◄──►│   Model Storage │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User          │    │   DeBERTa v3    │    │   DVC           │
│   Interface     │    │   + LoRA        │    │   Versioning    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 CI/CD Pipeline

The project includes two GitHub Actions workflows:

1. **Deploy to AWS ECR** (`deploy.yml`):
   - Builds Docker image with AWS credentials
   - Creates ECR repository if it doesn't exist
   - Pushes to ECR with commit-specific and latest tags
   - Triggers on push to main/master

2. **Deploy to EC2** (`deploy-ec2.yml`):
   - Uses AWS Systems Manager (SSM) for secure deployment
   - Stops and removes existing container
   - Cleans up old Docker images
   - Pulls latest image from ECR
   - Starts new container with proper environment variables
   - Runs after successful ECR deployment
   - Verifies deployment status

## 🧪 Testing

```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage

# Test local build
make test-local

# Test health endpoint
curl -f http://localhost:8501/_stcore/health

# Test production endpoint
curl -f http://your-ec2-ip/_stcore/health
```

## 📝 API Endpoints

- **Health Check**: `GET /_stcore/health`
- **Main App**: `GET /` (Streamlit interface)
- **Production URL**: `http://your-ec2-ip` (port 80)

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Restart the Docker container
2. **Model Not Found**: Ensure DVC credentials are correct
3. **AWS Access Denied**: Check IAM permissions for S3 and ECR
4. **EC2 Connection Failed**: Verify EC2 instance ID and SSM agent
5. **No Space Left on Device**: Increase EBS volume size (minimum 20GB recommended)
6. **Container Won't Start**: Check environment variables and logs
7. **App Not Accessible**: Verify security group allows port 80

### Logs

```bash
# View container logs
make logs

# Or directly with Docker
docker logs sentiment-analysis-app

# Check container status
docker ps

# Check system resources
df -h
free -h
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions, please open an issue in the GitHub repository.
