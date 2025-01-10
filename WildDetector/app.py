from flask import Flask, request, jsonify, Response
import os
import time
import threading
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Status variables
RUNNING_PORT = 5000
STATUS = 'idle'
SOURCE_HELP = 'http://10.200.3.28:5010/stream_raw'
SOURCE_CAMERA = 0   # Refer to '/dev/video0'
MODEL_FOLDER = 'models/'
MODEL = YOLO(f'{MODEL_FOLDER}Yolov11/yolo11m-pose.pt')
# For adhering to Flask's best practices TODO add all
app.config['STATUS'] = STATUS
app.config['SOURCE_CAMERA'] = SOURCE_CAMERA
app.config['MODEL_FOLDER'] = MODEL_FOLDER
# Ensure directories exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

def generate_frames(infer=False):
    cap = cv2.VideoCapture(SOURCE_CAMERA)

    if not cap.isOpened():
        print("Error: Unable to access the video feed. Is the raw stream active?")
        return

    # Initialize FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0  # Initialize fps to avoid reference before assignment

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Verify the settings
    current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    current_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Resolution: {current_width}x{current_height}")
    print(f"Frame Rate (FPS): {current_fps}")

    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while STATUS == "streaming_infer" if infer else "streaming_raw":
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video feed.")
            break

        # Get frame size
        height, width, _ = frame.shape

        # Update the frame count
        frame_count += 1

        # Calculate FPS every second
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Overlay FPS, image size, and camera FPS on the frame
        cv2.putText(frame, f"Image Size: {current_width}x{current_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Cam FPS: {current_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if infer:
            results = MODEL.predict(source=frame)
            frame = results[0].plot()

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def help_frames(infer=True):

    if SOURCE_HELP == 'none':
        print("None source provided!")
        return

    cap = cv2.VideoCapture(SOURCE_HELP)

    if not cap.isOpened():
        print("Error: Unable to access the video feed. Invalid source provided or raw stream active?")
        return

    # Initialize FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0  # Initialize fps to avoid reference before assignment

    # Get the settings
    current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    current_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Resolution: {current_width}x{current_height}")
    print(f"Frame Rate (FPS): {current_fps}")

    while infer:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video feed.")
            break

        # Get frame size
        height, width, _ = frame.shape

        # Update the frame count
        frame_count += 1

        # Calculate FPS every second
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Overlay FPS, image size, and camera FPS on the frame
        cv2.putText(frame, f"Image Size: {current_width}x{current_height}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Cam FPS: {current_fps:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if infer:
            results = MODEL.predict(source=frame)
            frame = results[0].plot()

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/stream_raw")   # tutta via il controllo dello stato e solo quando inizia, se gia partitio non ferma il flusso faw
def stream_raw():
    global STATUS
    if STATUS != "streaming_raw":
        return "Raw stream not active", 400
    return Response(generate_frames(infer=False), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_infer")
def stream_infer():
    global STATUS
    if STATUS != "streaming_infer":
        return "Inferred stream not active", 400
    return Response(generate_frames(infer=True), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_help")
def streaming_help():
    global STATUS
    if STATUS != "streaming_help":
        return "Inferred help stream not active", 400
    return Response(help_frames(infer=True), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_env", methods=["POST"])
def set_env():
    value = request.form.get("value")
    if value:
        os.environ["SOURCE_HELP"] = value
        SOURCE_HELP = value
        return f"SOURCE_HELP set to {value}", 200
    return "Invalid value", 400

@app.route("/change_status", methods=["POST"])
def change_status():
    global STATUS
    status = request.form.get("status")
    if status not in ["idle", "streaming_raw", "streaming_infer", "streaming_help"]:
        return "Invalid status", 400

    # Stop existing streaming threads
    STATUS = "idle"  # Force stopping current streams
    threading.Event().wait(1)

    if status == "streaming_raw":
        STATUS = "streaming_raw"
        threading.Thread(target=generate_frames, args=(5001, False)).start()
    elif status == "streaming_infer":
        STATUS = "streaming_infer"
        threading.Thread(target=generate_frames, args=(5002, True)).start()
    elif status == "streaming_help":
        STATUS = "streaming_help"
        threading.Thread(target=help_frames, args=(5002, True)).start()
    else:
        STATUS = "idle"

    return f"Status changed to {STATUS}", 200

@app.route("/")
def home():
    return """
    <h1>Flask UltraLytics Control Panel</h1>
    <form action="/set_env" method="POST">
        <label>Set SOURCE_HELP:</label>
        <input type="text" name="value">
        <button type="submit">Set</button>
    </form>
    <form action="/change_status" method="POST">
        <label>Change Status:</label>
        <select name="status">
            <option value="idle">Idle</option>
            <option value="streaming_raw">Stream Raw</option>
            <option value="streaming_infer">Stream Inferred</option>
            <option value="streaming_help">Help Inferred</option>
        </select>
        <button type="submit">Change</button>
    </form>
    <p>Current Status: {}</p>
    """.format(STATUS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=RUNNING_PORT)
