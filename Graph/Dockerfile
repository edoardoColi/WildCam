FROM python:3.9-slim

RUN pip install --no-cache-dir flask matplotlib requests

COPY . /app

WORKDIR /app

CMD ["python3", "app.py"]
