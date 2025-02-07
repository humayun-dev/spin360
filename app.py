from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO,SAM

# Initialize Flask application
app = Flask(__name__)

# Setup directories
UPLOAD_FOLDER = "uploads/"
FRAMES_FOLDER = "assets/images/frames/"
PROCESSED_FOLDER = "assets/images/processed/"
YOLO_MODEL_PATH = "/spin360/last.pt"
SAM_MODEL_PATH = "/spin360/sam2.1_b.pt"


# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Function to extract and sort frames from a video
def extract_frames_sorted(video_path, frames_folder, interval=10):
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    saved_frame_count = 1

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return False

    print("Extracting and renaming frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % interval == 0:
            frame_path = os.path.join(frames_folder, f"frame{saved_frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {frame_no} as {frame_path}.")
            saved_frame_count += 1

        frame_no += 1

    cap.release()
    print("Frame extraction and renaming complete.")
    return True if saved_frame_count > 1 else False

# Function to apply a grey color tint with 90% grey and 10% transparency
def apply_black_tint(image, mask):
    tinted_image = image.copy()
    mask = (mask * 255).astype(np.uint8)
    black_color = np.array([0, 0, 0])

    for c in range(3):
        tinted_image[:, :, c] = np.where(
            mask == 255,
            0.85 * black_color[c] + 0.15 * image[:, :, c],
            image[:, :, c]
        )
    return tinted_image

# Function to apply blur to the segmented regions (for plate)
def apply_blur(image, mask):
    blurred_image = image.copy()
    mask = (mask * 255).astype(np.uint8)
    blurred_image[mask == 255] = cv2.GaussianBlur(blurred_image[mask == 255], (99, 99), 30)
    return blurred_image

# Function to apply edge detection
def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=10, threshold2=150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

# Function to normalize car's distance by resizing and placing on a white canvas
def normalize_car_distance(image, mask, canvas_size=(2048, 1080), offset_x=0, offset_y=50, scale=1.2, reference_size=None):
    mask = (mask * 255).astype(np.uint8)
    car_pixels = cv2.bitwise_and(image, image, mask=mask)
    y_indices, x_indices = np.where(mask > 0)

    if len(x_indices) > 0 and len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        cropped_car = car_pixels[y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

        car_height, car_width = cropped_car.shape[:2]
        car_size = max(car_width, car_height)

        if reference_size is None:
            reference_size = car_size

        scale_factor = reference_size / car_size
        new_car_width = int(car_width * scale_factor)
        new_car_height = int(car_height * scale_factor)

        cropped_car = cv2.resize(cropped_car, (new_car_width, new_car_height), interpolation=cv2.INTER_CUBIC)
        cropped_mask = cv2.resize(cropped_mask, (new_car_width, new_car_height), interpolation=cv2.INTER_NEAREST)

        canvas_width, canvas_height = canvas_size
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

        x_offset = max(0, (canvas_width - new_car_width) // 2 + offset_x)
        y_offset = max(0, (canvas_height - new_car_height) // 2 + offset_y)

        x_offset = min(x_offset, canvas_width - new_car_width)
        y_offset = min(y_offset, canvas_height - new_car_height)

        for c in range(3):
            canvas[y_offset:y_offset + new_car_height, x_offset:x_offset + new_car_width, c] = np.where(
                cropped_mask > 0,
                cropped_car[:, :, c],
                canvas[y_offset:y_offset + new_car_height, x_offset:x_offset + new_car_width, c]
            )

        return canvas, reference_size
    else:
        print("No valid car mask found.")
        return image, reference_size

# Function to add ellipse shadow below the car
def add_shadow_to_car(image, car_mask, shadow_offset=(10, 5), shadow_scale=1.5):
    y_indices, x_indices = np.where(car_mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        print("No valid car mask found for shadow.")
        return image  # Return original if no mask

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    center_x = (x_max + x_min) // 2
    center_y = y_max + shadow_offset[1]
    axes = (int((x_max - x_min) // 2 * shadow_scale), int((y_max - y_min) // 4 * shadow_scale))

    shadow_image = np.zeros_like(image, dtype=np.uint8)
    cv2.ellipse(shadow_image, (center_x, center_y), axes, 0, 0, 180, (0, 0, 0), -1)
    shadow_image = cv2.GaussianBlur(shadow_image, (21, 21), 0)

    return cv2.addWeighted(image, 1, shadow_image, 0.5, 0)

# Process frames with depth and individual tints
def process_frames_with_depth_and_individual_tints(frames_folder, processed_folder):
    # Load YOLO and SAM models
    yolo_model = YOLO(YOLO_MODEL_PATH)
    sam_model = SAM(SAM_MODEL_PATH)

    os.makedirs(processed_folder, exist_ok=True)

    frame_files = [f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))]
    frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  # Sort by frame number

    if not frame_files:
        print("No frames found for processing.")
        return False

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        image = cv2.imread(frame_path)
        image_with_edges = apply_edge_detection(image)

        # Perform YOLO object detection
        results = yolo_model(image_with_edges)

        class_names = results[0].names
        car_class_id = None

        for k, v in class_names.items():
            if v.lower() == "car":
                car_class_id = k
                break

        if car_class_id is None:
            continue

        car_boxes = []
        for idx, cls in enumerate(results[0].boxes.cls):
            if int(cls) == int(car_class_id):
                bbox = results[0].boxes.xyxy[idx].tolist()
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                car_boxes.append((bbox, area))

        if car_boxes:
            largest_car_box = max(car_boxes, key=lambda x: x[1])[0]
            sam_results_car = sam_model(image_with_edges, bboxes=np.array([largest_car_box]), verbose=False, save=False, device="cpu")
            car_masks = [result.masks.data.cpu().numpy() for result in sam_results_car]

            for masks in car_masks:
                for mask in masks:
                    processed_image, _ = normalize_car_distance(image, mask, canvas_size=(2048, 1080))
                    processed_image_with_shadow = add_shadow_to_car(processed_image, mask)

                    output_path = os.path.join(processed_folder, f"processed_{frame_file}")
                    cv2.imwrite(output_path, processed_image_with_shadow)

    return True


# Combine processed frames into a video
def combine_frames_to_video(frames_folder, output_video_path):
    frame_files = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))])

    if not frame_files:
        print("No processed frames found to combine into video.")
        return False

    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video.write(frame)
    video.release()
    return True


# API endpoint to process video
@app.route('/process_video', methods=['POST'])
def process_video_api():
    if 'video' not in request.files:
        return jsonify({"error": "No video part in the request"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400

    # Save the video to disk
    video_path = os.path.join(UPLOAD_FOLDER, "input_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    # Extract frames from the video
    frames_extracted = extract_frames_sorted(video_path, FRAMES_FOLDER, interval=20)
    if not frames_extracted:
        return jsonify({"error": "Failed to extract frames from video"}), 500

    # Process the frames
    frames_processed = process_frames_with_depth_and_individual_tints(FRAMES_FOLDER, PROCESSED_FOLDER)
    if not frames_processed:
        return jsonify({"error": "Failed to process frames"}), 500

    # Combine processed frames into a video
    output_video_path = os.path.join(PROCESSED_FOLDER, "output_video.mp4")
    video_created = combine_frames_to_video(PROCESSED_FOLDER, output_video_path)
    if not video_created:
        return jsonify({"error": "Failed to combine frames into video"}), 500

    # Provide the processed video file for download
    return jsonify({
        "message": "Processing completed successfully",
        "download_url": f"/download/{os.path.basename(output_video_path)}"
    })


# Endpoint to serve processed video for download
@app.route('/download/<filename>', methods=['GET'])
def download_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
