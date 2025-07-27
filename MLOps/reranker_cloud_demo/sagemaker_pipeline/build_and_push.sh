cat > build_and_push.sh << 'EOF'
set -e

REGION="eu-north-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPOSITORY_NAME="crossencoder-sagemaker"
IMAGE_TAG="latest"

IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "Building image: ${IMAGE_URI}"

aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} --region ${REGION} || \
aws ecr create-repository --repository-name ${REPOSITORY_NAME} --region ${REGION}

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

docker build -t ${REPOSITORY_NAME}:${IMAGE_TAG} .

docker tag ${REPOSITORY_NAME}:${IMAGE_TAG} ${IMAGE_URI}

docker push ${IMAGE_URI}

echo "Image pushed successfully: ${IMAGE_URI}"
EOF
