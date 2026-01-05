FROM python:3.11-slim

WORKDIR /usr/local/app

RUN pip install onnx onnxruntime numpy fastapi uvicorn

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]