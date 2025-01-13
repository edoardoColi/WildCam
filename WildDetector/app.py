import os
import cv2
import json
import time
import torch
import datetime
import threading
import numpy as np
from torchvision import transforms
import torchvision
from ultralytics import YOLO
from collections import defaultdict
from flask import Flask, request, jsonify, Response, redirect # type: ignore

app = Flask(__name__)

# Status variables
RUNNING_PORT = int(os.getenv('RUNNING_PORT', 5000))     # Default to 5000 if not set
STATUS = 'get_raw'                                         # Default camera
SOURCE_CAMERA = 0                                       # Refer to '/dev/video0'
SOURCE_DATA = '/dev/video0'                                    # Customize for the inference source
MODEL_FOLDER = 'models/'
MODEL = YOLO(f'{MODEL_FOLDER}Yolo/yolo11m.pt')          # print(f"Layer {i}: {layer}") for i, layer in enumerate(MODEL.model.model)
BACKBONE = MODEL.model.model[0]                         # Backbone part of the model
NECK = MODEL.model.model[1]                             # Neck part of the model
HEAD = MODEL.model.model[2]                             # Head part of the model
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

        # Convert the result data dictionary to JSON string
        result_json = json.dumps(rewritten_res)

        # Yield the JSON-encoded results
        yield (b'--frame\r\n'
               b'Content-Type: application/json\r\n\r\n' + result_json.encode('utf-8') + b'\r\n')
        
    cap.release()

###
#   OPERATIVE PARTS (as helping device)
###

### with torch.no_grad():
### cis a ontext manager in PyTorch that disables gradient computation, which is useful during inference or evaluation.
### When used can save memory and computational resources, during model evaluation or inference you don’t need gradients since you’re not updating any parameters.

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

        # Resize frame to the desired input size
        frame_resized = cv2.resize(frame, (640, 640))
        
        # Convert to RGB (YOLO typically expects RGB)
        frame_rgb = frame_resized[..., ::-1]  # BGR to RGB
        
        # Normalize the image (YOLO uses values in range [0, 1])
        frame_normalized = frame_rgb / 255.0
        
        # Convert to Tensor and add batch dimension
        transform = transforms.ToTensor()
        frame_tensor = transform(frame_normalized).unsqueeze(0)  # Add batch dimension
        frame_tensor = frame_tensor.to(torch.double)  # Convert to torch.Double

        # Pass through the backbone
        backbone_out = MODEL.model.model[:10](frame_tensor.to(torch.double))  # Backbone layers
        # Pass through neck (if applicable)
        neck_out = MODEL.model.model[10:20](backbone_out)  # Neck layers
        # Pass through head (prediction layers)
        head_out = MODEL.model.model[20:](neck_out)  # Head layers

        head_out = head_out[head_out[..., 4] > 0.5]
        boxes = head_out[..., :4]
        scores = head_out[..., 4] * head_out[..., 5]  # Objectness * class score
        class_ids = head_out[..., 5].argmax(dim=-1)
        keep = torchvision.ops.nms(boxes, scores, 0.45)
        final_boxes = boxes[keep]
        final_scores = scores[keep]
        final_class_ids = class_ids[keep]

        print(final_boxes.shape)
        print(final_scores.shape)
        print(final_class_ids.shape)
        break

        # backbone_layers = MODEL.model.model[:10]  # Layers 0 to 9 for the backbone
        # neck_layers = MODEL.model.model[10:20]   # Layers 10 to 19 for the neck
        # head_layers = MODEL.model.model[20:]     # Layers 20 onwards for the head

        # backbone_out = backbone_layers(input_tensor)
        # neck_out = neck_layers(backbone_out)
        # results = head_layers(neck_out)


        # # Pre-process the frame to match YOLO input size
        # input_tensor = cv2.resize(frame, (640, 640))
        # input_tensor = input_tensor[..., ::-1]                            # Convert BGR to RGB
        # input_tensor = np.copy(input_tensor)                              # Create a copy of the array to avoid negative strides
        # input_tensor = np.transpose(input_tensor, (2, 0, 1))              # Change to (C, H, W)
        # input_tensor = np.expand_dims(input_tensor, axis=0)               # Add batch dimension
        # input_tensor = torch.from_numpy(input_tensor).float() / 255.0     # Normalize to [0, 1]

        # # Stage 1: Backbone (feature extraction)
        # with torch.no_grad():
        #     features = BACKBONE(input_tensor)           # Get the feature maps from the Backbone

        # # Stage 2: Neck (Feature Refinement)
        # with torch.no_grad():
        #     refined_features = NECK(features)

        # # Stage 3: Head (Final Predictions)
        # with torch.no_grad():
        #     predictions = HEAD(refined_features)        # Output tensors from the YOLO models seem to be feature maps of shape [1, C, H, W]

        # # Post-process the predictions, converting predictions into bounding boxes, confidences, and class IDs
        # #-----------------------------------------------------------------------------------------------------#

        # # print(MODEL)
        # # print(MODEL.model)
        # # print(MODEL.model.model)
        # print(MODEL.model[0])
        # # print(MODEL.model.model[1])
        # # print(MODEL.model.model[2])
        # # print(MODEL.model.model[3])
        # print("poi")

        # print(features.shape)
        # print(refined_features.shape)
        # print(predictions.shape)
        # print(predictions[0].shape)
        # break

        # # # Parse predictions (YOLO format: [x, y, w, h, conf, class1, class2, ...])
        # # predictions = predictions[0]  # Batch size is 1; take the first (and only) batch
        # # boxes, confidences, class_probs = predictions[:, :4], predictions[:, 4], predictions[:, 5:]

        # # # Flatten boxes and scores to handle the batch dimension properly
        # # boxes = boxes.reshape(-1, 4)  # Flatten the boxes to shape [num_predictions, 4]
        # # confidences = confidences.flatten()  # Flatten the scores to shape [num_predictions]
        # # class_probs = class_probs.flatten()  # Flatten the classes to shape [num_predictions]

        # # # Filter predictions by confidence threshold
        # # mask = confidences > 0.5
        # # boxes, confidences, class_probs = boxes[mask], confidences[mask], class_probs[mask]
        
        # # # Compute class probabilities and class IDs
        # # class_ids = torch.argmax(class_probs, dim=1)
        # # scores = confidences * torch.max(class_probs, dim=1).values

        # # # Apply Non-Maximum Suppression (NMS)
        # # keep_indices = nms(boxes, scores, IOU_THRESH)
        # # boxes, class_ids, scores = boxes[keep_indices], class_ids[keep_indices], scores[keep_indices]

        # # # Draw results on the frame
        # # for box, class_id, score in zip(boxes, class_ids, scores):
        # #     x, y, w, h = box
        # #     x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        # #     label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
        # #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  
  
        # # # Apply sigmoid to object confidence and class probabilities
        # # predictions[..., 4:] = torch.sigmoid(predictions[..., 4:])  # objectness and class scores
        
        # # # Get class probabilities and objectness score
        # # object_confidence = predictions[..., 4]  # Objectness score for each cell
        # # class_probs = predictions[..., 5:]  # Class probabilities
        
        # # # Apply threshold to object confidence
        # # object_confidence_mask = object_confidence > 0.5  # Apply confidence threshold
        # # predictions = predictions[object_confidence_mask]
        
        # # # Extract class labels
        # # class_preds = torch.argmax(class_probs, dim=-1)  # Get the predicted class for each detected object
        # # labels = class_preds.tolist()  # Convert to list for easier use
        
        # # print(labels)



        # # results = MODEL.postprocess(predictions[0])     # Apply Non-Maximum Suppression (NMS) and other filters
        
        # # Extract the bounding boxes, scores, and class labels
        # boxes = predictions[0][:, :4]  # Bounding boxes (x1, y1, x2, y2)
        # scores = predictions[0][:, 4]  # Confidence scores
        # classes = predictions[0][:, 5]  # Class IDs
        # print("Shape of boxes:", boxes.shape)
        # print("Shape of scores:", scores.shape)
        # print("Shape of classes:", classes.shape)

        # # Flatten boxes and scores to handle the batch dimension properly
        # boxes = boxes.reshape(-1, 4)  # Flatten the boxes to shape [num_predictions, 4]
        # scores = scores.flatten()  # Flatten the scores to shape [num_predictions]
        # classes = classes.flatten()  # Flatten the classes to shape [num_predictions]

        # # Apply a confidence threshold (optional, e.g., keep detections with confidence > 0.5)
        # confidence_threshold = 0.5
        # mask = scores > confidence_threshold

        # # Filter predictions based on the confidence threshold
        # filtered_boxes = boxes[mask]
        # filtered_scores = scores[mask]
        # filtered_classes = classes[mask]

        # # mask = confidences > 0.5
        # # boxes, confidences, class_probs = boxes[mask], confidences[mask], class_probs[mask]
        # print("Shape of filtered_classes:", filtered_classes.shape)

        # # class_ids = torch.argmax(filtered_classes, dim=1)
        # # scores = filtered_scores * torch.max(filtered_classes, dim=1).values

        # # # Apply Non-Maximum Suppression (NMS)
        # # keep_indices = nms(boxes, scores, IOU_THRESH)
        # # boxes, class_ids, scores = boxes[keep_indices], class_ids[keep_indices], scores[keep_indices]

        # # # Draw results on the frame
        # # for box, class_id, score in zip(boxes, class_ids, scores):
        # #     x, y, w, h = box
        # #     x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        # #     label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
        # #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # # Create the results object, which usually works in this format:
        # # (boxes, scores, classes, frame) in the expected format
        # results = [YOLO.Results(frame, boxes=filtered_boxes, scores=filtered_scores, classes=filtered_classes)]

        frame = results[0].plot()   # used to overlay the results (such as bounding boxes, class labels, or other annotations) onto the image

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
