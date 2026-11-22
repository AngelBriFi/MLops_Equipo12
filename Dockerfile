FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
COPY mlflow.db .
COPY mlruns ./mlruns

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]