# Use a specific Python version
FROM python:3.9.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt to the container
COPY requirements.txt .

# Install dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 8080 to the outside world
EXPOSE 8080

# Command to run the app using Flask's built-in server
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]