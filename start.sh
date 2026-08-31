#!/bin/bash
# Render.com build & start script
# This runs before the app starts in production

echo "=== DeepFake Investigation System: Startup ==="

# Download model from Hugging Face if not present
python download_model.py

# Start Flask app with Gunicorn (production WSGI server)
exec gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --timeout 300 \
    --log-level info
