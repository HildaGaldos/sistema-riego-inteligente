FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt -r /app/ml/requirements.txt
COPY . /app
ENV PYTHONPATH=/app/ml/src
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
