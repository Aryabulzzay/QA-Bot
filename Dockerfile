FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8501
EXPOSE 8501 8000

CMD ["bash", "-c", "streamlit run app.py & uvicorn api:app --host 0.0.0.0 --port 8000"]
