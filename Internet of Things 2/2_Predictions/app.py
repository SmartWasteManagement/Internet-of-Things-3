# Import necessary libraries
from flask import Flask, render_template, Response
import cv2
from yolo_predictions import YOLO_Pred

# Create a Flask application instance
app = Flask(__name__)
# Instantiate YOLO_Pred object with the YOLO model and data YAML file paths
yolo = YOLO_Pred('./Model2/weights/best.onnx', 'data.yaml')
camera = None  # Initialize the camera object

# Function to generate frames for video streaming
def generate_frames():
    while True:
        if camera is not None:
            # Read a frame from the camera
            success, frame = camera.read()
            if not success:
                break
            else:
                # Resize frame for faster processing
                resized_frame = cv2.resize(frame, (640, 480))
                
                # Perform pothole detection on the resized frame
                frame_with_potholes = yolo.predictions(resized_frame)
                
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame_with_potholes)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video streaming

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to start the camera

@app.route('/start_camera')
def start_camera_route():
    return start_camera()

# Function to start the camera
def start_camera():
    global camera
    if camera is None:
        # Initialize the camera with a lower resolution and lower FPS
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640 pixels
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480 pixels
        camera.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate to 15 FPS
        print("Camera started successfully")
        return 'Camera started successfully'
    else:
        return 'Camera is already running'

# Route to stop the camera
@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release() # Release the camera object
        camera = None
    return 'Camera stopped successfully'

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
