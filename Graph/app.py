import os
import time
import json
import requests
import matplotlib.pyplot as plt
from collections import deque
from flask import Flask, Response, render_template_string # type: ignore

app = Flask(__name__)

RUNNING_PORT = int(os.getenv('RUNNING_PORT', 5020))     # Default to 5020 if not set
STREAM_URL = str(os.getenv('STREAM_URL', "http://10.200.3.28:5000/stream_inf"))

time_data = deque(maxlen=12)  # Stores 12 timestamps (5s intervals for 1 minute)
real_count_data = deque(maxlen=12)  # Stores real_count values for the same period

# This function generates the HTML content for the graph
def generate_html():
    # Generate the plot dynamically with the latest data
    fig, ax = plt.subplots()

    ax.plot(time_data, real_count_data, marker='o', color='b')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Real Count")
    ax.set_title("Real Count vs Time")

    # Format the x-axis to show time intervals
    ax.set_xticks(time_data)
    ax.set_xticklabels([f"{t:.1f}" for t in time_data])

    # Return the image as a response
    from io import BytesIO
    import base64
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_data = base64.b64encode(img_io.getvalue()).decode('utf8')
    plt.close(fig)
    
    # HTML to render the image on the page
    return f'<img src="data:image/png;base64,{img_data}"/>'

def stream_json_from_url(url):
    try:
        with requests.get(url, stream=True) as response:
            if response.status_code != 200:
                yield f"Failed to connect to '{STREAM_URL}': {response.status_code}\n"
                return

            buffer = ""
            for chunk in response.iter_lines():  # Removed decode_unicode=True
                if chunk:
                    # Decode the chunk from bytes to string before concatenating
                    buffer += chunk.decode("utf-8")

                    # Check if the buffer contains a complete JSON object
                    if buffer.startswith("--frame") and "Content-Type: application/json" in buffer:
                        try:
                            json_start = buffer.index("{")
                            json_data = json.loads(buffer[json_start:])
                            buffer = ""  # Clear buffer after processing the current frame

                            # Handle the JSON data
                            threshold = 0.5         # Confidence threshold
                            if "box_count" in json_data and "boxes" in json_data and "names" in json_data:
                                result = []
                                for obj_name, count in json_data["box_count"].items():
                                    real_count = count
                                    class_code = next((key for key, value in json_data["names"].items() if value == obj_name), "unknown")
                                    boxes = json_data["boxes"].get(str(class_code)+".0", [])
                                    confidences = [box[4] for box in boxes]  # Extract probabilities
                                    real_count -= sum(1 for conf in confidences if conf < threshold)

                                    result.append({"class": obj_name, "code": class_code, "count": count, "confidence": confidences, "real": real_count})
                                yield json.dumps(result) + "\n"

                        except (json.JSONDecodeError, ValueError):
                            continue

    except Exception as e:
        yield f"Error: {e}\n"

@app.route("/stream")
def stream():
    # Use Flask Response to stream the data
    def generate():
        yield from stream_json_from_url(STREAM_URL)
    return Response(generate(), content_type="text/plain")

@app.route("/")
def graph():
    # Generate the graph HTML and display it
    return render_template_string("""
        <html>
            <head>
                <title>Real Count vs Time</title>
            </head>
            <body>
                <h1>Real Count vs Time</h1>
                {{ graph|safe }}
                <br />
                <p>Refreshing every 5 seconds...</p>
            </body>
        </html>
    """, graph=generate_html())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=RUNNING_PORT)
