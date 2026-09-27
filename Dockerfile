FROM python:3.11-slim

# System deps for Pillow/scipy wheels build fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# checkpoints/ and history.db are created/written at runtime — mount them as
# volumes (see docker-compose.yml) so a trained model and analysis history
# persist across container restarts.
RUN mkdir -p checkpoints

EXPOSE 5000
ENV FLASK_ENV=production

CMD ["python", "app.py"]
