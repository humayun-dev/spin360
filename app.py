from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import uuid
from pathlib import Path
import tempfile
# Initialize FastAPI application
app = FastAPI()

# Persistent storage directory for processed videos
PROCESSED_STORAGE_DIR = "processed_videos"
os.makedirs(PROCESSED_STORAGE_DIR, exist_ok=True)

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
    # Convert boolean mask to uint8 before resizing
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = (mask * 255).astype(np.uint8)
    car_pixels = cv2.bitwise_and(image, image, mask=mask)
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) > 0 and len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        cropped_car = car_pixels[y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        if cropped_car.size == 0 or cropped_mask.size == 0:
            return image, reference_size
        car_height, car_width = cropped_car.shape[:2]
        car_size = max(car_width, car_height)
        if reference_size is None:
            reference_size = car_size
        scale_factor = reference_size / car_size
        new_car_width = int(car_width * scale_factor)
        new_car_height = int(car_height * scale_factor)
        if new_car_width <= 0 or new_car_height <= 0:
            return image, reference_size
        cropped_car = cv2.resize(cropped_car, (new_car_width, new_car_height), interpolation=cv2.INTER_CUBIC)
        cropped_mask = cv2.resize(cropped_mask, (new_car_width, new_car_height), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((*canvas_size[::-1], 3), 255, dtype=np.uint8)
        x_offset = max(0, min((canvas_size[0] - new_car_width) // 2 + offset_x, canvas_size[0] - new_car_width))
        y_offset = max(0, min((canvas_size[1] - new_car_height) // 2 + offset_y, canvas_size[1] - new_car_height))
        roi_height = min(new_car_height, canvas_size[1] - y_offset)
        roi_width = min(new_car_width, canvas_size[0] - x_offset)
        for c in range(3):
            canvas[y_offset:y_offset + roi_height, x_offset:x_offset + roi_width, c] = np.where(
                cropped_mask[:roi_height, :roi_width] > 0,
                cropped_car[:roi_height, :roi_width, c],
                canvas[y_offset:y_offset + roi_height, x_offset:x_offset + roi_width, c]
            )
        return canvas, reference_size
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
    yolo_model = YOLO("spin360/last.pt")
    sam_model = SAM("spin360/sam2.1_b.pt")
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

# Process video API
@app.post("/process_video")
async def process_video_api(video: UploadFile = File(...)):
    try:
        if not video:
            raise HTTPException(status_code=400, detail="No video file uploaded")

        # Create temporary directories for this request
        with tempfile.TemporaryDirectory() as upload_dir, \
             tempfile.TemporaryDirectory() as frames_dir, \
             tempfile.TemporaryDirectory() as processed_dir:

            # Save the video to a temporary directory
            video_path = os.path.join(upload_dir, "input_video.mp4")
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)

            # Extract frames
            frames_extracted = extract_frames_sorted(video_path, frames_dir, interval=20)
            if not frames_extracted:
                return JSONResponse(
                    status_code=422,
                    content={"message": "No frames could be extracted from the video"}
                )

            # Process frames
            frames_processed = process_frames_with_depth_and_individual_tints(frames_dir, processed_dir)
            if not frames_processed:
                return JSONResponse(
                    status_code=422,
                    content={"message": "No cars detected in video frames"}
                )

            # Combine frames into a video
            unique_filename = f"output_{uuid.uuid4()}.mp4"  # Unique filename
            output_video_path = os.path.join(PROCESSED_STORAGE_DIR, unique_filename)
            video_created = combine_frames_to_video(processed_dir, output_video_path)
            if not video_created:
                return JSONResponse(
                    status_code=422,
                    content={"message": "Could not create output video from processed frames"}
                )

            # Return the filename for downloading
            return JSONResponse({
                "message": "Processing completed successfully",
                "download_url": f"/download_and_delete/{unique_filename}"
            })

    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={"message": f"Video processing failed: {str(e)}"}
        )

# Download and delete endpoint
@app.get("/download_and_delete/{filename}")
async def download_and_delete_video(filename: str):
    try:
        # Locate the processed video in the persistent storage directory
        file_path = Path(PROCESSED_STORAGE_DIR) / filename

        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # Return the file and delete it after download
        response = FileResponse(
            file_path,
            media_type='application/octet-stream',
            filename=filename
        )
        # os.remove(file_path)  # Delete the file after download
        return response

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to download or delete file: {str(e)}")

@app.get("/delete_video/{filename}")
async def root(filename: str):
    try:
        # Locate the processed video in the persistent storage directory
        file_path = Path(PROCESSED_STORAGE_DIR) / filename

        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)
        return JSONResponse({
            "message": f"File {filename} deleted successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to delete file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)