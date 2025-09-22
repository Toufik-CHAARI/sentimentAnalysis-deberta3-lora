# Sentiment Analysis with DeBERTa v3 + LoRA

A Streamlit application for sentiment analysis using a fine-tuned DeBERTa v3 model with LoRA (Low-Rank Adaptation). The model is stored in S3 and pulled via DVC during Docker image building.

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
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## 🐳 Docker Deployment

### Local Docker Build

```bash
# Build the Docker image
make build

# Run the container locally
make run

# Or run in detached mode
make run-detached
```

### Available Make Commands

```bash
make help                    # Show all available commands
make build                   # Build Docker image locally
make run                     # Run Docker container locally
make run-detached           # Run Docker container in background
make stop                    # Stop running container
make logs                    # Show container logs
make shell                   # Open shell in running container
make clean                   # Clean up Docker resources
make test-local              # Build and test locally
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
- `DVC_BUCKET`: Your S3 bucket name for DVC
- `EC2_HOST`: Your EC2 instance public IP or domain
- `EC2_USERNAME`: SSH username for EC2 (usually `ubuntu`)
- `EC2_SSH_KEY`: Private SSH key for EC2 access

### Manual EC2 Deployment

If you prefer manual deployment:

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account-id.dkr.ecr.us-east-1.amazonaws.com

# Pull and run the image
docker pull your-account-id.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis-deberta3-lora:latest
docker run -d --name sentiment-analysis-app -p 80:8501 \
  -e AWS_ACCESS_KEY_ID=your-access-key \
  -e AWS_SECRET_ACCESS_KEY=your-secret-key \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e DVC_BUCKET=your-bucket \
  your-account-id.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis-deberta3-lora:latest
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

The model is configured in `app.py`:

- **Model Path**: `deberta3_lora_400000k/`
- **Labels**:
  - 0: "😞 Negative Sentiment"
  - 1: "😊 Positive Sentiment"

## 📊 Usage

1. **Text Input**: Enter text in the text area
2. **Analysis**: Click "Analyze Sentiment" to get predictions
3. **Visualization**: View sentiment distribution and confidence scores
4. **Batch Processing**: Upload CSV files for bulk analysis
5. **EDA**: Explore the dataset with interactive visualizations

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
   - Builds Docker image
   - Pushes to ECR
   - Triggers on push to main/master

2. **Deploy to EC2** (`deploy-ec2.yml`):
   - Deploys to EC2 instance
   - Runs after successful ECR deployment
   - Updates running container

## 🧪 Testing

```bash
# Test local build
make test-local

# Test health endpoint
curl -f http://localhost:8501/_stcore/health
```

## 📝 API Endpoints

- **Health Check**: `GET /_stcore/health`
- **Main App**: `GET /` (Streamlit interface)

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Restart the Docker container
2. **Model Not Found**: Ensure DVC credentials are correct
3. **AWS Access Denied**: Check IAM permissions for S3 and ECR
4. **EC2 Connection Failed**: Verify SSH key and security groups

### Logs

```bash
# View container logs
make logs

# Or directly with Docker
docker logs sentiment-analysis-deberta3-lora-container
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
