import os
import cv2
import json
import time
import torch
import datetime
import threading
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from collections import defaultdict
from ultralytics.engine.results import Results
from flask import Flask, request, jsonify, Response, redirect # type: ignore

app = Flask(__name__)

# Status variables
RUNNING_VIDEO = os.getenv('RUNNING_FLAG', 'false').lower() in ('true', '1', 't', 'yes', 'y')
RUNNING_PORT = int(os.getenv('RUNNING_PORT', 5000))     # Default to 5000 if not set
STATUS = 'idle'                                         # Default camera
SOURCE_CAMERA = 0                                       # Refer to '/dev/video0'
SOURCE_DATA = 'none'                                    # Customize for the inference source
MODEL_FOLDER = 'models/'
MODEL = YOLO(f'{MODEL_FOLDER}Yolo/yolo11n.pt')          # print(f"Layer {i}: {layer}") for i, layer in enumerate(MODEL.model.model)
BACKBONE = MODEL.model.model[0]                         # Backbone part of the model
NECK = MODEL.model.model[1]                             # Neck part of the model
HEAD = MODEL.model.model[2]                             # Head part of the model
# For adhering to Flask's best practices
app.config['RUNNING_VIDEO'] = RUNNING_VIDEO
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
    print(f"Camera Resolution: {current_width}x{current_height}")
    print(f"Camera Frame Rate: {current_fps}")

    # Initialize FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.perf_counter()
    while STATUS == "send_raw":
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video feed.")
            break   # Using continue allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume

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
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Verify the settings
    current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {current_width}x{current_height}")
    print(f"Frame Rate (FPS): {current_fps}")

    # Initialize FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.perf_counter()
    while STATUS == "send_inf":
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video feed.")
            break   # Using continue allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume

        results = MODEL.predict(source=frame)

        frame_count += 1                                        # Update the frame count
        if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            frame_count = 0                                     # Restart the counter
            start_time = time.perf_counter()                    # Restart the times

        extracted_res = {
            'inference_time': results[0].speed['inference'],        # Espressed in ms
            'boxes': results[0].boxes.data.tolist() if results[0].boxes else [],        # Contain: [x1, y1, x2, y2, confidence, class_id]
            # 'keypoints': results[0].keypoints.data.tolist() if results[0].keypoints else [],
            # 'masks': results[0].masks.data.tolist() if results[0].masks else [],
            'names': results[0].names,
            # 'path': results[0].path,
        }
        box_groups = defaultdict(list)                  # Dictionary of boxes grouped by class ID
        for box in extracted_res['boxes']:
            class_id = box[5]                           # The 6th value represents the class ID
            box_groups[class_id].append(box[:5])        # Append the box without the class ID
        rewritten_res = {
            'inference_time': extracted_res['inference_time'],
            'box_count': {extracted_res['names'][class_id]: len(boxes) for class_id, boxes in box_groups.items()},
            'boxes': box_groups,
            'yolo_fps': 1000/results[0].speed['inference'],
            'fps': fps,
            'names': extracted_res['names']
        }

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', results[0].plot())
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # # Convert the result data dictionary to JSON string
        # result_json = json.dumps(rewritten_res)

        # # Yield the JSON-encoded results
        # yield (b'--frame\r\n'
        #        b'Content-Type: application/json\r\n\r\n' + result_json.encode('utf-8') + b'\r\n')
        
    cap.release()

###
#   OPERATIVE PARTS (as helping device)
###

### with torch.no_grad():
### cis a ontext manager in PyTorch that disables gradient computation, which is useful during inference or evaluation.
### When used can save memory and computational resources, during model evaluation or inference you don’t need gradients since you’re not updating any parameters.

### print(MODEL.model)  or look at https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml for layer structure
### as HEAD output, referred as 'detectHead', we can find:
### detectHead[0]: Primary predictions (e.g., processed for object detection).
### detectHead[1]: Feature maps used for further calculations or analysis (e.g., visualization, debugging, additional model layers).
### Can examine better the content with:
###  if isinstance(b23, (list, tuple)):
###      for i, tensor in enumerate(b23):
###          print(f"Type of element {i} in b23: {type(tensor)}")
###          if isinstance(tensor, torch.Tensor):
###              print(f"Shape of tensor {i} in b23: {tensor.shape}")
###          elif isinstance(tensor, list):
###              print(f"Element {i} in b23 is a list with length: {len(tensor)}")
###          else:
###              print(f"Unexpected type for element {i}: {type(tensor)}")
###  else:
###      print("Unexpected output type for b23:", type(b23))
###  if isinstance(b23, (list, tuple)):
###      for i, item in enumerate(b23):
###          if isinstance(item, list):
###              for j, sub_item in enumerate(item):
###                  print(f"Type of sub-item {j} in list at b23[{i}]: {type(sub_item)}")
###                  if isinstance(sub_item, torch.Tensor):
###                      print(f"Shape of sub-item {j}: {sub_item.shape}")

def consume_raw():
    cap = cv2.VideoCapture(SOURCE_DATA)

    if not cap.isOpened():
        print(f"Error: Unable to access the video feed from '{SOURCE_DATA}'. Is the stream active elsewhere?")
        return

    while STATUS in ["get_raw", "get_back", "get_neck", "get_head"]:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video feed.")
            break   # Using 'continue' allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume

        # Pre-process the frame to match YOLO input size
        frame_resized = cv2.resize(frame, (640, 640))               # Resize frame to the desired input size
        frame_rgb = frame_resized[..., ::-1]                        # Convert BGR to RGB (YOLO typically expects RGB)
        frame_normalized = frame_rgb / 255.0                        # Normalize the image (YOLO uses values in range [0, 1])
        transform = transforms.ToTensor()                           # Convert to Tensor and add batch dimension
        frame_tensor = transform(frame_normalized).unsqueeze(0)     # Add batch dimension
        frame_tensor = frame_tensor.to(torch.float32)               # Convert to float32 as the first layer require

        # Can also use this to pre-process
        # input_tensor = cv2.resize(frame, (640, 640))
        # input_tensor = input_tensor[..., ::-1]                            # Convert BGR to RGB
        # input_tensor = np.copy(input_tensor)                              # Create a copy of the array to avoid negative strides
        # input_tensor = np.transpose(input_tensor, (2, 0, 1))              # Change to (C, H, W)
        # input_tensor = np.expand_dims(input_tensor, axis=0)               # Add batch dimension
        # input_tensor = torch.from_numpy(input_tensor).float() / 255.0     # Normalize to [0, 1]

        # Stage 1: Backbone (feature extraction)    # Input a torch.Size([1, 3, 640, 640]) !FOR YOLO11n.pt!
        b0 = MODEL.model.model[0](frame_tensor)     # Output a torch.Size([1, 16, 320, 320])
        b1 = MODEL.model.model[1](b0)               # Output a torch.Size([1, 32, 160, 160])
        b2 = MODEL.model.model[2](b1)               # Output a torch.Size([1, 64, 160, 160])
        b3 = MODEL.model.model[3](b2)               # Output a torch.Size([1, 64, 80, 80])
        b4 = MODEL.model.model[4](b3)               # Output a torch.Size([1, 128, 80, 80])
        b5 = MODEL.model.model[5](b4)               # Output a torch.Size([1, 128, 40, 40])
        b6 = MODEL.model.model[6](b5)               # Output a torch.Size([1, 128, 40, 40])
        b7 = MODEL.model.model[7](b6)               # Output a torch.Size([1, 256, 20, 20])
        b8 = MODEL.model.model[8](b7)               # Output a torch.Size([1, 256, 20, 20])

        # Stage 2: Neck (Feature Refinement)
        b9 = MODEL.model.model[9](b8)               # Output a torch.Size([1, 256, 20, 20])
        b10 = MODEL.model.model[10](b9)             # Output a torch.Size([1, 256, 20, 20])
        b11 = MODEL.model.model[11](b10)            # Output a torch.Size([1, 256, 40, 40])
        b12 = MODEL.model.model[12]([b11,b6])       # Output a torch.Size([1, 384, 40, 40])
        b13 = MODEL.model.model[13](b12)            # Output a torch.Size([1, 128, 40, 40])
        b14 = MODEL.model.model[14](b13)            # Output a torch.Size([1, 128, 80, 80])
        b15 = MODEL.model.model[15]([b14,b4])       # Output a torch.Size([1, 256, 80, 80])
        b16 = MODEL.model.model[16](b15)            # Output a torch.Size([1, 64, 80, 80])
        b17 = MODEL.model.model[17](b16)            # Output a torch.Size([1, 64, 40, 40])
        b18 = MODEL.model.model[18]([b17,b13])      # Output a torch.Size([1, 192, 40, 40])
        b19 = MODEL.model.model[19](b18)            # Output a torch.Size([1, 128, 40, 40])
        b20 = MODEL.model.model[20](b19)            # Output a torch.Size([1, 128, 20, 20])
        b21 = MODEL.model.model[21]([b20,b10])      # Output a torch.Size([1, 384, 20, 20])
        b22 = MODEL.model.model[22](b21)            # Output a torch.Size([1, 256, 20, 20])

        # Stage 3: Head (Final Predictions)
        b23 = MODEL.model.model[23]([b16,b19,b22])

        # Post-process the predictions, converting predictions into bounding boxes, confidences, and class IDs
        # TODO

        # predictions = b23[0]
        # boxes = []
        # for pred in predictions[0]:  # Assuming batch size is 1
        #     x_center, y_center, w, h = pred[:4]  # Box coordinates
        #     score = pred[4]  # Objectness score
        #     class_scores = pred[5:]  # Class scores
        #     class_id = torch.argmax(class_scores)
        #     final_score = score * class_scores[class_id]
        #     # Convert to [x1, y1, x2, y2] format
        #     x1 = x_center - w / 2
        #     y1 = y_center - h / 2
        #     x2 = x_center + w / 2
        #     y2 = y_center + h / 2
        #     print([x1, y1, x2, y2, final_score, class_id])
        #     # Append as [x1, y1, x2, y2, conf, cls]
        #     boxes.append([x1, y1, x2, y2, final_score, class_id])
        # # Convert to tensor
        # boxes_tensor = torch.tensor(boxes)  # Shape: [num_detections, 6]
        # # print(MODEL.names)
        # my_dict = {i: str(i) for i in range(10001)}
        # results = Results(path=None, orig_img=frame, boxes=boxes_tensor, names=my_dict)
        # frame = results[0].plot()

        # Convert the result data to JSON string
        result_json = json.dumps(b23[0].tolist())

        # Yield the JSON-encoded results
        yield (b'--frame\r\n'
               b'Content-Type: application/json\r\n\r\n' + result_json.encode('utf-8') + b'\r\n')

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
    if STATUS not in ["get_raw", "get_back", "get_neck", "get_head"]:
        return "Head stream not active", 400
    return Response(generate_head(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_inf")
def stream_inf():
    global STATUS
    if STATUS != "send_inf":
        return "Inference stream not active", 400
    return Response(generate_inf(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/help_raw")
def help_raw():
    global STATUS
    if STATUS != "get_raw":
        return "HELP Raw stream not active", 400
    return Response(consume_raw(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/help_back")
def help_back():
    global STATUS
    if STATUS != "get_back":
        return "HELP Backbone stream not active", 400
    return Response(consume_back(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/help_neck")
def help_neck():
    global STATUS
    if STATUS != "get_neck":
        return "HELP Neck stream not active", 400
    return Response(consume_neck(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/help_head")
def help_head():
    global STATUS
    if STATUS != "get_head":
        return "HELP Head stream not active", 400
    return Response(consume_head(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/change_status", methods=["POST"])
def change_status():
    global STATUS
    ols_stat = STATUS
    status = request.form.get("status")
    if status not in ["idle", "send_raw", "send_back", "send_neck", "send_head", "send_inf", "get_raw", "get_back", "get_neck", "get_head"]:
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
    elif status == "get_raw":
        STATUS = "get_raw"
    elif status == "get_back":
        STATUS = "get_back"
    elif status == "get_neck":
        STATUS = "get_neck"
    elif status == "get_head":
        STATUS = "get_head"
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
    <h1>Flask Wild Detector Control Panel</h1>
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
            <option value="none">----------------</option>
            <option value="get_raw">Help in Raw</option>
            <option value="get_back">Help in Backbone</option>
            <option value="get_neck">Help in Neck</option>
            <option value="get_head">Help in Head</option>
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
