FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraud_api.py .
COPY model.joblib .

EXPOSE 8000

CMD ["uvicorn", "fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]
