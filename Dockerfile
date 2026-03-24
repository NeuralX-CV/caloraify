FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt

COPY app.py app.py

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]