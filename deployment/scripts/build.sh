#!/bin/bash

# Build script for UDL Rating Framework deployment
set -e

# Configuration
IMAGE_NAME="udl-rating-framework"
IMAGE_TAG="${1:-latest}"
REGISTRY="${REGISTRY:-localhost:5000}"
DOCKERFILE_PATH="deployment/docker/Dockerfile"

echo "Building UDL Rating Framework Docker image..."
echo "Image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build the Docker image
docker build \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
    --target production \
    .

echo "Build completed successfully!"

# Optionally push to registry
if [ "${PUSH_TO_REGISTRY}" = "true" ]; then
    echo "Pushing image to registry..."
    docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "Push completed!"
fi

# Run security scan if trivy is available
if command -v trivy &> /dev/null; then
    echo "Running security scan..."
    trivy image "${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo "Image build process completed!"
echo "To run locally: docker run -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"