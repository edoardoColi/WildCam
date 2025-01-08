import os
import json
import requests
from flask import Flask, request, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import uuid
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
CUSTOM_MODEL_FOLDER = 'static/custom_models/'
MODEL_FOLDER = 'models/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CUSTOM_MODEL_FOLDER'] = CUSTOM_MODEL_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CUSTOM_MODEL_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

current_model = None
camera_on = False
input_source = None
cap = None
is_image = False
fps_value = 0  # Global variable to store FPS
loading = 0
results_store = []  # Store results globally

# Define available YOLO models by version
model_versions = {
    'Yolov5': ['yolov5nu', 'yolov5su', 'yolov5mu', 'yolov5lu', 'yolov5xu'],
    'Yolov8': [
        'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
        'yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg',
        'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose',
        'yolov8n-obb', 'yolov8s-obb', 'yolov8m-obb', 'yolov8l-obb', 'yolov8x-obb',
        'yolov8n-cls', 'yolov8s-cls', 'yolov8m-cls', 'yolov8l-cls', 'yolov8x-cls'
    ],
    'Yolov11': [
        'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
        'yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg', 'yolo11x-seg',
        'yolo11n-pose', 'yolo11s-pose', 'yolo11m-pose', 'yolo11l-pose', 'yolo11x-pose',
        'yolo11n-obb', 'yolo11s-obb', 'yolo11m-obb', 'yolo11l-obb', 'yolo11x-obb',
        'yolo11n-cls', 'yolo11s-cls', 'yolo11m-cls', 'yolo11l-cls', 'yolo11x-cls'
    ]
}

def download_model(model_path):
    # 获取模型文件的目录（不包括文件名）
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 仅当模型文件不存在时下载
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found, downloading...")
        model_name = os.path.basename(model_path)
        # 如果是 YOLOv8，则可以直接下载
        if 'yolov8' in model_name:
            YOLO(model_name)
        else:
            print(f"Please manually download {model_name} and place it in {model_dir}")
    else:
        print(f"Model {model_path} already exists, skipping download.")

    # 返回加载的模型对象
    return YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html', model_versions=model_versions)

@app.route('/loading_status')
def loading_status():
    global loading
    return jsonify({'loading': loading})

@app.route('/detect', methods=['POST'])
def detect():
    global current_model, camera_on, input_source, cap, is_image, results_store
    model_version = request.form.get('version')
    model_choice = request.form.get('model')
    input_type = request.form.get('input_type')
    tensorrt_enabled = request.form.get('tensorrt') == 'true'  # Check TensorRT option

    if model_choice == 'custom':
        # 用户选择了自定义模型
        custom_model_file = request.files.get('custom_model')
        if custom_model_file:
            custom_model_filename = f"{uuid.uuid4()}_{custom_model_file.filename}"
            custom_model_path = os.path.join(app.config['CUSTOM_MODEL_FOLDER'], model_version, custom_model_filename)
            os.makedirs(os.path.dirname(custom_model_path), exist_ok=True)
            custom_model_file.save(custom_model_path)
            print(f"Custom model uploaded to {custom_model_path}")
            model_path = custom_model_path  # Use the custom model
        else:
            return jsonify({'error': 'No custom model file uploaded'})
    else:
        # 使用预定义的模型
        if model_version and model_choice:
            model_path = os.path.join(app.config['MODEL_FOLDER'], model_version, f"{model_choice}.pt")
        else:
            return jsonify({'error': 'No valid version or model selected'})

    # 检查并下载模型（如果需要）
    if model_choice != 'custom':
        current_model = download_model(model_path)
    else:
        current_model = YOLO(model_path)

    # 处理 TensorRT 转换
    if tensorrt_enabled:
        engine_path = model_path.replace('.pt', '.engine')
        if os.path.exists(engine_path):
            current_model = YOLO(engine_path)
        else:
            print(f"Exporting model {model_path} to TensorRT format...")
            model = YOLO(model_path)
            model.export(format='engine', device=0)  # 指定设备
            while not os.path.exists(engine_path):
                time.sleep(1)
            print(f"TensorRT engine exported and saved to {engine_path}")
            current_model = YOLO(engine_path)
    else:
        current_model = YOLO(model_path)

    # Reset state for new detection
    if cap is not None:
        cap.release()
        cap = None
    input_source = None
    camera_on = False
    is_image = False

    # Handle input types (image, video, webcam)
    if input_type == 'image':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            input_source = filepath
            is_image = True
            return jsonify({'image': True})
    elif input_type == 'video':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            input_source = filepath
            camera_on = True
            cap = cv2.VideoCapture(input_source)
            return jsonify({'video': True})
    elif input_type == 'webcam':
        input_source = 0
        camera_on = True
        cap = cv2.VideoCapture(input_source)
        return jsonify({'webcam': True})

    return jsonify({'error': 'Invalid input type'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_on, cap
    camera_on = False
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({'stopped': True})

def generate_frames(model):
    global camera_on, input_source, cap, is_image, fps_value, loading, results_store
    prev_frame_time = 0
    new_frame_time = 0

    if is_image:
        img = cv2.imread(input_source)
        results = model(img)
        loading = 1
        for result in results:
            annotated_frame = result.plot()
            results_store = {
                'inference_time': result.speed['inference'],
                'boxes': result.boxes.data.tolist() if result.boxes else [],
                'keypoints': result.keypoints.data.tolist() if result.keypoints else [],
                'masks': result.masks.data.tolist() if result.masks else [],
                'names': result.names,
                'path': result.path,
            }
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    while camera_on:
        success, frame = cap.read()
        if not success:
            break
        new_frame_time = time.time()
        results = model(frame)
        loading = 1
        for result in results:
            annotated_frame = result.plot()
            results_store = {
                'inference_time': result.speed['inference'],
                'boxes': result.boxes.data.tolist() if result.boxes else [],
                'keypoints': result.keypoints.data.tolist() if result.keypoints else [],
                'masks': result.masks.data.tolist() if result.masks else [],
                'names': result.names,
                'path': result.path,
            }
            fps_value = 1000/result.speed['inference']
            prev_frame_time = new_frame_time

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(current_model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def get_results():
    global results_store, fps_value
    results = {
        'results': results_store,
        'fps': fps_value,
    }
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
