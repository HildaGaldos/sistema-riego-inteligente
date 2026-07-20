FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt -r /app/ml/requirements.txt streamlit
COPY . /app
ENV PYTHONPATH=/app/ml/src
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
