FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# evercompliant POC patch: upstream ships dev mode (--reload watches the FS and forks
# a reloader process) — wrong for a deployed image.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
