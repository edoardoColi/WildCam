FROM ultralytics/ultralytics:latest-jetson-jetpack6

RUN pip install --no-cache-dir flask pika

COPY . /usr/src/ultralytics

WORKDIR /usr/src/ultralytics/

CMD ["python3", "app.py"]
