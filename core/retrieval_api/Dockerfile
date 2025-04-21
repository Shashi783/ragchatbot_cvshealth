FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY core/ core/
# Copy the rest of the application
# COPY . .

# Copy requirements first to leverage Docker cache
COPY core/retrieval_api/requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "core.retrieval_api.app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"] 