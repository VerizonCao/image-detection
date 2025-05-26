# Use AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Install build tools and required libs for InsightFace and OpenCV
RUN yum install -y \
    gcc \
    gcc-c++ \
    make \
    cmake \
    python3-devel \
    mesa-libGL \
    mesa-libGLU \
    libXext \
    libXrender \
    libpng \
    libjpeg-turbo \
    zlib \
    freetype \
    && yum clean all


# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}
# WORKDIR /app

# Copy requirements.txt and wheels directory
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# get wheel
RUN pip install --upgrade setuptools wheel

# Install all standard packages from PyPI (excluding insightface)
RUN pip install -r requirements.txt


# Copy application code
COPY app.py .
COPY .env.local .env.local

# Set Lambda function entrypoint
CMD ["app.handler"]
