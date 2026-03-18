FROM python:3.10-slim

# Install system dependencies required by RDKit's Cairo/X11 rendering backend
# These are the libraries that were missing in your original deployment:
# - libxrender1, libxext6: X11 rendering (needed by Cairo)
# - libexpat1: XML parsing (needed for SVG generation)
# - libcairo2: Cairo graphics library
# - libfreetype6: Font rendering
# - libfontconfig1: Font configuration
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libexpat1 \
    libcairo2 \
    libfreetype6 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for OpenShift compatibility
# OpenShift runs containers as arbitrary UIDs in the root group
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app && \
    chown -R appuser:0 /app && \
    chmod -R g=u /app

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY wsgi.py .
COPY DiffAlign ./DiffAlign
COPY evaluation ./evaluation
RUN pip install --no-cache-dir -e ./DiffAlign 

# OpenShift runs as arbitrary UID but in root group (GID 0)
# Make sure the app directory is writable by root group
RUN chgrp -R 0 /app && chmod -R g=u /app

# Use non-privileged port (OpenShift doesn't allow ports < 1024)
EXPOSE 8080

# Run as non-root user
USER 1001

# Start gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "600", "wsgi:application"]
