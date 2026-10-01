# Intraday Strategies portal - Linux image for the VPS (see README "Running on the VPS").
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IS_DATA_DIR=/app/var \
    TZ=Asia/Kolkata

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 1000 portal && mkdir -p /app/var && chown -R portal /app
USER portal

EXPOSE 8100
# bound to 0.0.0.0, so the portal refuses to start without PORTAL_PASSWORD
CMD ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8100"]
