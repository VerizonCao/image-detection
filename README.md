# Face Detection Lambda Function

This repository contains a Lambda function for face detection and image content moderation using InsightFace and Google's Gemini API.

## Prerequisites

- Docker
- AWS CLI configured with appropriate permissions
- Python 3.10 or later (for local testing)

## Environment Variables

Create a `.env` file with the following variables:

```
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
S3_BUCKET_NAME=your_bucket_name

# Database Configuration
PGHOST=your_db_host
PGDATABASE=your_db_name
PGUSER=your_db_user
PGPASSWORD=your_db_password

# Google API
GOOGLE_API_KEY=your_gemini_api_key
```

## Building and Deploying

1. Build the Docker image:
   ```bash
   docker build -t face-detection-lambda .
   ```

2. Create an ECR repository (if not exists):
   ```bash
   aws ecr create-repository --repository-name face-detection-lambda
   ```

3. Login to ECR:
   ```bash
   aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-account-id.dkr.ecr.your-region.amazonaws.com
   ```

4. Tag and push the image:
   ```bash
   docker tag face-detection-lambda:latest your-account-id.dkr.ecr.your-region.amazonaws.com/face-detection-lambda:latest
   docker push your-account-id.dkr.ecr.your-region.amazonaws.com/face-detection-lambda:latest
   ```

5. Create a new Lambda function in AWS Console:
   - Choose "Container image" as the deployment package
   - Select the image you just pushed to ECR
   - Configure the following settings:
     - Memory: At least 1024MB (recommended 2048MB)
     - Timeout: At least 30 seconds
     - Environment variables: Add all variables from your .env file

## Local Testing

Run the test script:
```bash
python test_handler.py
```

## Function Input Format

The Lambda function expects an event with the following structure:
```json
{
    "img_path": "rita-avatars/image.jpg",  // S3 path or local path
    "avatar_id": "unique_avatar_id"        // Required for S3 images
}
```

## Notes

- For local testing, the function will skip database updates
- S3 paths must start with "rita-avatars/"
- The function performs both face detection and content moderation
- Results are stored in the avatar_moderation table for S3 images