FROM python:3.13.7-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.13.7-slim AS runner
WORKDIR /app
COPY --from=builder /app/.venv .venv
ENV PATH="/app/.venv/bin:$PATH"
COPY . .
RUN useradd -m -u 8888 appuser && chown -R appuser:appuser /app /home/appuser
USER appuser
EXPOSE 8000
CMD ["fastapi", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]