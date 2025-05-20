FROM python:3.10

# --- Configuración básica del entorno ---
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 
ENV PYTHONUNBUFFERED=1

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["celery", "-A", "tasks", "worker", "-Q", "tracker_queue", "--loglevel=info"]
