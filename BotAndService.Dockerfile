FROM python:3.10-slim

WORKDIR /bot_and_service

COPY req.txt req.txt

RUN pip3 install --upgrade pip setuptools setuptools_rust Cython && pip3 install -r req.txt

COPY . .


CMD python main.py >> /var/log/rag_bot.log 2>&1 & uvicorn fast_api_service:app --host 0.0.0.0 --port 81 >> /var/log/rag_service.log 2>&1
