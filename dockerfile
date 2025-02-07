# Use a specific Python version
FROM python:3.9.13-slim

# Install system dependencies for OpenCV, Gunicorn, and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0  # Correct package for libgthread

# Set the working directory in the container to the root directory (where all files are)
WORKDIR /spin360

# Copy all the application files into the container's working directory
COPY . /spin360

# Install dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r /spin360/requirements.txt

# Expose port 8080 to the outside world
EXPOSE 8080

# Install Gunicorn for serving the app
RUN pip install gunicorn

# Command to run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
