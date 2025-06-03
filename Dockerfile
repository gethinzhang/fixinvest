# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements_live.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_live.txt

# Copy application code
COPY hi5.py .
COPY live_trader_offline.py .
COPY gcp_config.json .
#COPY smtp_config.json .
#COPY gcp-credentials.json .

# Create directory for credentials
RUN mkdir -p /app/credentials

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York

# Create a non-root user
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Command to run the application
CMD ["python", "live_trader_offline.py", "--engine", "gcp", "--email", "zgxcassar@gmail.com"] 