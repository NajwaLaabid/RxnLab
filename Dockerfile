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

# Install Python dependencies.
# Torch must land BEFORE the PyG packages (torch-scatter / torch-sparse / etc.):
# those packages don't declare torch as a build dep but import it at the top of
# setup.py to detect build flags. With modern pip's resolver, all packages are
# collected before any install runs, so if torch-scatter falls through to a
# source build (no matching wheel for the platform), `import torch` fails.
# Installing torch first sidesteps that whole class of build-order errors.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY wsgi.py .
COPY app ./app
COPY DiffAlign ./DiffAlign
COPY evaluation ./evaluation
RUN pip install --no-cache-dir -e ./DiffAlign

# OpenShift runs as arbitrary UID but in root group (GID 0)
# Make sure the app directory is writable by root group
RUN chgrp -R 0 /app && chmod -R g=u /app

# OpenShift runs as an arbitrary UID with no home → HOME defaults to "/" (unwritable),
# which breaks every lib that caches under $HOME: dgl (/.dgl, fatal for LocalRetro),
# syntheseus' checkpoint cache, matplotlib, gunicorn. Point them all at /app (g=u writable).
ENV HOME=/app \
    MPLCONFIGDIR=/app/.config/matplotlib \
    SYNTHESEUS_CACHE_DIR=/app/.cache/syntheseus

# Use non-privileged port (OpenShift doesn't allow ports < 1024)
EXPOSE 8080

# Run as non-root user
USER 1001

# Start gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "600", "wsgi:application"]
