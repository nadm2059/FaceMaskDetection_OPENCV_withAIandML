# Use lightweight Python 3.10 slim base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Upgrade pip and copy dependency definition
RUN python -m pip install --upgrade pip
COPY requirements.txt .

# Install Python dependencies without caching wheels
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code and models into container
COPY . .

# Launch app.py from the app/ directory
CMD ["python", "app/app.py"]
