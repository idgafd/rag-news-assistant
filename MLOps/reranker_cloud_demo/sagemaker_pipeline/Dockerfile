FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY requirements-sagemaker.txt .

RUN pip install --no-cache-dir -r requirements-sagemaker.txt

# environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

RUN mkdir -p /opt/ml/processing/input
RUN mkdir -p /opt/ml/processing/output
RUN mkdir -p /tmp

CMD ["python3"]