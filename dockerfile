# Use a specific Python version
FROM python:3.9.13-slim

# Install system dependencies for OpenCV, Gunicorn, and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0  # Correct package for libgthread

# Install gsutil to download models from Google Cloud Storage
RUN apt-get install -y gsutil

# Set the working directory in the container
WORKDIR /spin360

# Download models from Google Cloud Storage
RUN gsutil cp gs://spin360models/last.pt /spin360/last.pt
RUN gsutil cp gs://spin360models/sam2.1_b.pt /spin360/sam2.1_b.pt

# Copy the rest of the application files into the container
COPY . /spin360

# Install dependencies from the requirements.txt file
COPY requirements.txt /spin360
RUN pip install --no-cache-dir -r /spin360/requirements.txt

# Expose port 8080 to the outside world
EXPOSE 8080

# Install Gunicorn
RUN pip install gunicorn

# Command to run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
