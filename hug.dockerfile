FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y git

# Set workdir
WORKDIR /code

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Run FastAPI (Hugging Face default port is 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
