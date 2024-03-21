FROM python:3.10-slim

WORKDIR /fast_api_service

COPY req.txt req.txt

RUN pip3 install --upgrade pip setuptools setuptools_rust Cython && pip3 install -r req.txt

COPY . .

CMD uvicorn fast_api_service:app --host 0.0.0.0 --port 80 >> /var/log/rag_service.log 2>&1
