# Start with a base image that contains Python (or the framework you need)
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the contents of your AI project folder into the container
COPY . /app

# Install any dependencies (assuming you have a requirements.txt file in the AI folder)
RUN pip install --no-cache-dir -r requirements.txt

# Expose any necessary ports (if your AI service needs it)
EXPOSE 8018

# Set the default command to run your AI script or app
CMD ["python", "scripts/api.py"]
