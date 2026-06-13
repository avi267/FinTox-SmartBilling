FROM python:3.11-slim

WORKDIR /app

# Install dependencies first — separate layer for caching
COPY fintox_smartbilling_v3/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY fintox_smartbilling_v3/ ./fintox_smartbilling_v3/

WORKDIR /app/fintox_smartbilling_v3

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
