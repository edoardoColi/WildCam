import os
import cv2
import json
import time
import datetime
import threading
from ultralytics import YOLO
from flask import Flask, request, jsonify, Response, redirect

app = Flask(__name__)

# Status variables
RUNNING_PORT = int(os.getenv('RUNNING_PORT', 5000))     # Default to 5000 if not set
STATUS = 'idle'                                         # Default camera
SOURCE_CAMERA = 0                                       # Refer to '/dev/video0'
SOURCE_DATA = 'none'                                    # Customize for the inference source
MODEL_FOLDER = 'models/'
MODEL = YOLO(f'{MODEL_FOLDER}Yolov11/yolo11m.pt')
# For adhering to Flask's best practices
app.config['RUNNING_PORT'] = RUNNING_PORT
app.config['STATUS'] = STATUS
app.config['SOURCE_CAMERA'] = SOURCE_CAMERA
app.config['SOURCE_DATA'] = SOURCE_DATA
app.config['MODEL_FOLDER'] = MODEL_FOLDER
# Ensure directories exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

###
#   OPERATIVE PARTS (as capable device)
###

### yield is a keyword in Python that allows a function to return a value and pause its execution, so that it can later resume where it left off

def generate_raw():
    cap = cv2.VideoCapture(SOURCE_CAMERA)
    if not cap.isOpened():
        print(f"Error: Unable to access the video feed from '{SOURCE_CAMERA}'. Is the stream active elsewhere?")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Verify the settings
    current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {current_width}x{current_height}")
    print(f"Frame Rate: {current_fps}")

    # Initialize FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.perf_counter()
    while STATUS == "send_raw":
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video feed.")
            break

        frame_count += 1                                        # Update the frame count
        if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            frame_count = 0                                     # Restart the counter
            start_time = time.perf_counter()                    # Restart the times

        # Overlay FPS, image size, and camera FPS on the frame
        cv2.putText(frame, f"Image Size: {current_width}x{current_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)    # The triplete (x, x, x) is the color of the text
        cv2.putText(frame, f"Frame Rate: {current_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f"Estimated FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            (frame.shape[1] - cv2.getTextSize(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0][0] - 10, 
             frame.shape[0] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def generate_back():
    return
    
def generate_neck():
    return
    
def generate_head():
    return
    
def generate_inf():
    cap = cv2.VideoCapture(SOURCE_CAMERA)

    if not cap.isOpened():
        print(f"Error: Unable to access the video feed from '{SOURCE_CAMERA}'. Is the stream active elsewhere?")
        return

    # Initialize FPS calculation
    frame_count = 0
    fps = 0  # Initialize fps to avoid reference before assignment
    start_time = time.time()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Verify the settings
    current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {current_width}x{current_height}")
    print(f"Frame Rate (FPS): {current_fps}")

    while STATUS == "send_inf":
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video feed.")
            break

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

        results = MODEL.predict(source=frame)
        print(results[0])
        frame = results[0].plot()   # used to overlay the results (such as bounding boxes, class labels, or other annotations) onto the image

        cv2.putText(frame, f"FPS da yolo: {1000/results[0].speed['inference']:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        results_store = {
            'inference_time': results[0].speed['inference'],
            'boxes': results[0].boxes.data.tolist() if results[0].boxes else [],
            'keypoints': results[0].keypoints.data.tolist() if results[0].keypoints else [],
            'masks': results[0].masks.data.tolist() if results[0].masks else [],
            'names': results[0].names,
            'path': results[0].path,
        }
        fps_value = 1000/results[0].speed['inference']
        share_result = {
            'results': results_store,
            'fps': fps_value,
        }

        # Convert the result data dictionary to JSON string
        result_json = json.dumps(share_result)

        # Yield the JSON-encoded results
        yield (b'--frame\r\n'
               b'Content-Type: application/json\r\n\r\n' + result_json.encode('utf-8') + b'\r\n')
        
    cap.release()

###
#   OPERATIVE PARTS (as helping device)
###

def generate_help():
    cap = cv2.VideoCapture(SOURCE_DATA)

    if not cap.isOpened():
        print(f"Error: Unable to access the video feed from '{SOURCE_DATA}'. Is the stream active elsewhere?")
        return

    # Initialize FPS calculation
    frame_count = 0
    fps = 0  # Initialize fps to avoid reference before assignment
    start_time = time.time()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Verify the settings
    current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {current_width}x{current_height}")
    print(f"Frame Rate (FPS): {current_fps}")

    while STATUS == "send_help":
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video feed.")
            break

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

        results = MODEL.predict(source=frame)
        print(results[0])
        frame = results[0].plot()   # used to overlay the results (such as bounding boxes, class labels, or other annotations) onto the image

        cv2.putText(frame, f"FPS da yolo: {1000/results[0].speed['inference']:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        results_store = {
            'inference_time': results[0].speed['inference'],
            'boxes': results[0].boxes.data.tolist() if results[0].boxes else [],
            'keypoints': results[0].keypoints.data.tolist() if results[0].keypoints else [],
            'masks': results[0].masks.data.tolist() if results[0].masks else [],
            'names': results[0].names,
            'path': results[0].path,
        }
        fps_value = 1000/results[0].speed['inference']
        share_result = {
            'results': results_store,
            'fps': fps_value,
        }

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def consume_back():
    return
    
def consume_neck():
    return
    
def consume_head():
    return

###
#   ENDPOINTS
###

### mimetype specifies the media type of the HTTP response:
### - multipart/x-mixed-replace is used for server push content, where the server continuously sends new parts
### - boundary=frame defines the boundary string that separates individual parts of the data stream

@app.route("/stream_raw")
def stream_raw():
    global STATUS
    if STATUS != "send_raw":
        return "Raw stream not active", 400
    return Response(generate_raw(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_back")
def stream_back():
    global STATUS
    if STATUS != "send_back":
        return "Backbone stream not active", 400
    return Response(generate_back(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_neck")
def stream_neck():
    global STATUS
    if STATUS != "send_neck":
        return "Neck stream not active", 400
    return Response(generate_neck(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_head")
def stream_head():
    global STATUS
    if STATUS != "send_head":
        return "Head stream not active", 400
    return Response(generate_head(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_inf")
def stream_inf():
    global STATUS
    if STATUS != "send_inf":
        return "Inference stream not active", 400
    return Response(generate_inf(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_help")
def stream_help():
    global STATUS
    if STATUS != "send_help":
        return "Helping stream not active", 400
    return Response(generate_help(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/change_status", methods=["POST"])
def change_status():
    global STATUS
    ols_stat = STATUS
    status = request.form.get("status")
    if status not in ["idle", "send_raw", "send_back", "send_neck", "send_head", "send_inf", "send_help"]:
        print("Invalid status")
        return redirect("/")

    STATUS = "idle"             # To stop previous streaming threads tasks
    threading.Event().wait(1)   # introduces a 1-second delay

    if status == "send_raw":
        STATUS = "send_raw"
    elif status == "send_back":
        STATUS = "send_back"
    elif status == "send_neck":
        STATUS = "send_neck"
    elif status == "send_head":
        STATUS = "send_head"
    elif status == "send_inf":
        STATUS = "send_inf"
    elif status == "send_help":
        STATUS = "send_help"
    else:
        STATUS = "idle"

    return f"Status changed from '{ols_stat}' to '{STATUS}'", 200

@app.route("/set_env", methods=["POST"])
def set_env():
    global SOURCE_DATA  # Recall global variable to update it
    old_val = SOURCE_DATA
    value = request.form.get("value")
    if value:
        SOURCE_DATA = value
        return f"SOURCE_DATA set from '{old_val}' to '{value}'", 200
    return "Invalid value, can't be empty", 400

@app.route("/")
def home():
    return """
    <h1>Flask UlWild Detector Control Panel</h1>
    <form action="/set_env" method="POST">
        <label>Set SOURCE_DATA, actually '{source_data}':</label>
        <input type="text" name="value">
        <button type="submit">Set</button>
    </form>
    <form action="/change_status" method="POST">
        <label>Change Status:</label>
        <select name="status">
            <option value="idle">Idle</option>
            <option value="send_raw">Share Raw</option>
            <option value="send_back">Share Backbone</option>
            <option value="send_neck">Share Neck</option>
            <option value="send_head">Share Head</option>
            <option value="send_inf">Share Inference</option>
            <option value="send_help">Help in Inference</option>
        </select>
        <button type="submit">Change</button>
    </form>
    <p>Current Status: {status}</p>
    """.format(source_data=SOURCE_DATA, status=STATUS)

###
#   MAIN
###
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=RUNNING_PORT)
