#!/bin/bash

# Build the Docker image
docker build -t face-detection-lambda .

echo "Docker image built successfully. You can now push it to ECR and use it in Lambda."
echo "To push to ECR, you'll need to:"
echo "1. Create an ECR repository (if not exists)"
echo "2. Tag the image with your ECR repository URI"
echo "3. Push the image to ECR"
echo ""
echo "Example commands:"
echo "aws ecr create-repository --repository-name face-detection-lambda"
echo "aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-account-id.dkr.ecr.your-region.amazonaws.com"
echo "docker tag face-detection-lambda:latest your-account-id.dkr.ecr.your-region.amazonaws.com/face-detection-lambda:latest"
echo "docker push your-account-id.dkr.ecr.your-region.amazonaws.com/face-detection-lambda:latest" 