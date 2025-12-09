# FROM python:3.10-slim
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PIP_ROOT_USER_ACTION=ignore

# Set working directory
WORKDIR /app

# Install system dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        bash \
        supervisor \
    && rm -rf /var/lib/apt/lists/* \
              /var/cache/debconf/* \
              /tmp/* \
              /var/tmp/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for logs and data
RUN mkdir -p /tmp/logs /app/data/chroma_db && \
    chmod 755 /tmp/logs /app/data

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app /tmp/logs
USER app

# Expose Flask (5000) and Streamlit (8501) ports
EXPOSE 5000
EXPOSE 8501

# Health check for Flask with longer timeout
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start both Flask + Streamlit via supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
