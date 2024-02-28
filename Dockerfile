FROM python:3.10-slim

WORKDIR /gp_tg_bt

COPY req.txt req.txt

RUN pip3 install --upgrade pip setuptools setuptools_rust Cython && pip3 install -r req.txt

COPY . .

CMD ["python", "main.py"]
