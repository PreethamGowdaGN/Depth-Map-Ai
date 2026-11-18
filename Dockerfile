# One-command run:  docker build -t depthmap-ai . && docker run -p 7860:7860 depthmap-ai
# Also used as-is by Hugging Face Spaces (Docker SDK, port 7860).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:7860/_stcore/health')"
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
