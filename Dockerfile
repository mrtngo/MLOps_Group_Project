# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Set the PYTHONPATH to include the project's source code
ENV PYTHONPATH=/code

# Copy requirements files first to leverage Docker's layer caching
COPY requirements.txt requirements-dev.txt requirements-api.txt ./

# Install Python dependencies
# Using --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r requirements-dev.txt -r requirements-api.txt

# Copy the application code and necessary assets into the container
COPY app ./app
COPY src ./src
COPY conf ./conf
COPY models ./models

# Expose the port the app runs on
EXPOSE 8000

# Add a health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 