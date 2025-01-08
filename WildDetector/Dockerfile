FROM ultralytics/ultralytics:latest-jetson-jetpack6

WORKDIR /usr/src/ultralytics/

RUN pip install --no-cache-dir flask

COPY . .

CMD ["python3", "app.py"]
