from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import cv2
import numpy as np
import json
from ultralytics import YOLO, SAM
import uuid
from pathlib import Path
import tempfile
import shutil

# Persistent storage directory for processed videos
PROCESSED_STORAGE_DIR = "processed_videos"
os.makedirs(PROCESSED_STORAGE_DIR, exist_ok=True)

app = FastAPI()

# ----- Helper Functions from your new code -----

def rotate_image_with_padding(image, angle):
    """Rotate image without cropping by adding padding."""
    height, width = image.shape[:2]
    diagonal = int(np.sqrt(height**2 + width**2))
    padded_image = cv2.copyMakeBorder(
        image,
        (diagonal - height) // 2, (diagonal - height) // 2,
        (diagonal - width) // 2, (diagonal - width) // 2,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    padded_h, padded_w = padded_image.shape[:2]
    center = (padded_w // 2, padded_h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (padded_w, padded_h))
    return rotated_image

def extract_corrected_frames(video_path, rotations, corrected_folder):
    """
    Extract frames from the video using rotation correction.
    Expects `rotations` as a list of dicts with keys "timestamp" (ms) and "x" (tilt angle).
    Only every 20th entry is used.
    """
    os.makedirs(corrected_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_counter = 1
    for idx, entry in enumerate(rotations):
        if idx % 20 != 0:
            continue
        timestamp_ms = entry["timestamp"]
        x_tilt = entry["x"]
        # Calculate frame number based on timestamp (in seconds)
        frame_number = int((timestamp_ms / 1000.0) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame at {timestamp_ms} ms not found.")
            continue
        # Correct tilt (note the negative sign as in your code)
        corrected_frame = rotate_image_with_padding(frame, -x_tilt)
        corrected_frame = cv2.resize(corrected_frame, (1100, 1100))
        corrected_filename = os.path.join(corrected_folder, f"frame_{frame_counter}.jpg")
        cv2.imwrite(corrected_filename, corrected_frame)
        frame_counter += 1
    cap.release()
    print("Corrected frames saved successfully.")

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

def apply_blur(image, mask):
    blurred_image = image.copy()
    mask = (mask * 255).astype(np.uint8)
    blurred_region = image.copy()
    blurred_region[mask == 255] = cv2.GaussianBlur(image[mask == 255], (99, 99), 30)
    return blurred_region

def apply_edge_detection(image):
    bilateral_filtered = cv2.bilateralFilter(image, d=7, sigmaColor=70, sigmaSpace=70)
    gray = cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 30:
            cv2.drawContours(edges, [contour], -1, 0, thickness=cv2.FILLED)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.GaussianBlur(edges, (5, 5), 2)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

def normalize_car_distance(image, mask, background_path="background.png", offset_x=0, offset_y=70, scale=1.5, reference_size=None):
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
        scale_factor = (reference_size / car_size) * scale
        new_car_width = int(car_width * scale_factor)
        new_car_height = int(car_height * scale_factor)
        cropped_car = cv2.resize(cropped_car, (new_car_width, new_car_height), interpolation=cv2.INTER_CUBIC)
        cropped_mask = cv2.resize(cropped_mask, (new_car_width, new_car_height), interpolation=cv2.INTER_NEAREST)
        background = cv2.imread(background_path)
        if background is None:
            print("Error: Background image not found. Using white canvas instead.")
            background = np.full((1080, 2048, 3), 255, dtype=np.uint8)
        else:
            background = cv2.resize(background, (2048, 1080))
        x_offset = max(0, (2048 - new_car_width) // 2 + offset_x)
        y_offset = max(0, (1080 - new_car_height) // 2 + offset_y)
        x_offset = min(x_offset, 2048 - new_car_width)
        y_offset = min(y_offset, 1080 - new_car_height)
        for c in range(3):
            background[y_offset:y_offset + new_car_height, x_offset:x_offset + new_car_width, c] = np.where(
                cropped_mask > 0,
                cropped_car[:, :, c],
                background[y_offset:y_offset + new_car_height, x_offset:x_offset + new_car_width, c]
            )
        return background, reference_size
    else:
        print("No valid car mask found.")
        return image, reference_size

def process_frames_with_depth_and_individual_tints(frames_folder, processed_folder, yolo_model_path, sam_model_path):
    yolo_model = YOLO(yolo_model_path)
    sam_model = SAM(sam_model_path)
    os.makedirs(processed_folder, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.jpg', '.png'))],
                         key=lambda x: int(''.join(filter(str.isdigit, x))))
    if not frame_files:
        print("No frames found for processing.")
        return False
    print("Processing frames with edge detection, car detection, and individual region tinting...")
    region_classes = ["back", "front", "windb", "windf"]
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        image = cv2.imread(frame_path)
        image_with_edges = apply_edge_detection(image)
        results = yolo_model(image_with_edges)
        class_names = results[0].names
        # Get class ids for regions, plate and car
        region_class_ids = {region: next((k for k, v in class_names.items() if v == region), None) for region in region_classes}
        plate_class_id = next((k for k, v in class_names.items() if v == "plate"), None)
        car_class_id = next((k for k, v in class_names.items() if v == "car"), None)
        region_boxes = {region: [] for region in region_classes}
        plate_boxes = []
        car_boxes = []
        for idx, cls in enumerate(results[0].boxes.cls):
            for region in region_classes:
                if int(cls) == region_class_ids.get(region):
                    region_boxes[region].append(results[0].boxes.xyxy[idx].tolist())
            if plate_class_id is not None and int(cls) == plate_class_id:
                plate_boxes.append(results[0].boxes.xyxy[idx].tolist())
            if car_class_id is not None and int(cls) == car_class_id:
                bbox = results[0].boxes.xyxy[idx].tolist()
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                car_boxes.append((bbox, area))
        # Apply tinting for region boxes
        for region in region_classes:
            boxes = region_boxes.get(region, [])
            if boxes:
                sam_results_region = sam_model(image_with_edges, bboxes=np.array(boxes), verbose=False, save=False, device="cpu")
                region_masks = [result.masks.data.cpu().numpy() for result in sam_results_region]
                for masks in region_masks:
                    for mask in masks:
                        image = apply_black_tint(image, mask)
        # Apply blur for plate boxes
        if plate_boxes:
            sam_results_plate = sam_model(image_with_edges, bboxes=np.array(plate_boxes), verbose=False, save=False, device="cpu")
            plate_masks = [result.masks.data.cpu().numpy() for result in sam_results_plate]
            for masks in plate_masks:
                for mask in masks:
                    image = apply_blur(image, mask)
        # Process car region
        if car_boxes:
            largest_car_box = max(car_boxes, key=lambda x: x[1])[0]
            sam_results_car = sam_model(image_with_edges, bboxes=np.array([largest_car_box]), verbose=False, save=False, device="cpu")
            car_masks = [result.masks.data.cpu().numpy() for result in sam_results_car]
            for masks in car_masks:
                for mask in masks:
                    processed_image, _ = normalize_car_distance(
                        image, mask, background_path="background.png", offset_x=0, offset_y=70, scale=1.5
                    )
                    output_path = os.path.join(processed_folder, f"processed_frame_{frame_file}")
                    cv2.imwrite(output_path, processed_image)
                    print(f"Processed and saved: {output_path}")
        else:
            print(f"No cars detected in frame {frame_file}.")
    print("All frames processed.")
    return True

def images_to_video(image_folder, output_video, frame_rate=10):
    """
    Convert images from a folder into a video.
    """
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png", ".jpeg"))],
                    key=lambda x: int(''.join(filter(str.isdigit, x))))
    if not images:
        print("No images found in the folder.")
        return None
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        video.write(frame)
    video.release()
    print(f"Video saved as {output_video}")
    return output_video

# ----- FastAPI Endpoints -----

@app.post("/process_video")
async def process_video(
    video: UploadFile = File(...),
    rotations: str = Form(...)
):
    """
    Expects:
      - a video file upload,
      - a form field 'rotations' containing a JSON string (list of dicts with keys "timestamp" and "x")
    """
    try:
        # Parse rotations JSON from form field
        rotations_data = json.loads(rotations)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rotations JSON: {e}")

    try:
        # Create temporary directories for this request
        with tempfile.TemporaryDirectory() as upload_dir, \
             tempfile.TemporaryDirectory() as frames_dir, \
             tempfile.TemporaryDirectory() as processed_dir:
            
            # Save the uploaded video
            video_path = os.path.join(upload_dir, "input_video.mp4")
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)
            
            # Step 1: Extract corrected frames using the provided rotations data
            extract_corrected_frames(video_path, rotations_data, frames_dir)
            
            # Step 2: Process the frames using YOLO and SAM models.
            # (Update the model paths as needed.)
            yolo_model_path = "spin360/lastnew.pt"
            sam_model_path = "spin360/sam2.1_l.pt"
            processed = process_frames_with_depth_and_individual_tints(frames_dir, processed_dir, yolo_model_path, sam_model_path)
            if not processed:
                return JSONResponse(
                    status_code=422,
                    content={"message": "No cars detected in video frames"}
                )
            
            # Step 3: Combine processed frames into a video.
            temp_video_path = os.path.join(upload_dir, "temp_output.mp4")
            video_created = images_to_video(processed_dir, temp_video_path, frame_rate=10)
            if not video_created:
                return JSONResponse(
                    status_code=422,
                    content={"message": "Could not create output video from processed frames"}
                )
            
            # Move the output video to persistent storage with a unique filename
            unique_filename = f"output_{uuid.uuid4()}.mp4"
            final_video_path = os.path.join(PROCESSED_STORAGE_DIR, unique_filename)
            shutil.move(temp_video_path, final_video_path)
            
            # Return the download URL (the /download_and_delete endpoint will serve and then delete the file)
            return JSONResponse({
                "message": "Processing completed successfully",
                "download_url": f"/download_and_delete/{unique_filename}"
            })
    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={"message": f"Video processing failed: {str(e)}"}
        )

@app.get("/download_and_delete/{filename}")
async def download_and_delete_video(filename: str):
    try:
        file_path = Path(PROCESSED_STORAGE_DIR) / filename
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        response = FileResponse(
            file_path,
            media_type='application/octet-stream',
            filename=filename
        )
        # Optionally, uncomment the next line to delete after download.
        # os.remove(file_path)
        return response
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to download or delete file: {str(e)}")

@app.get("/delete_video/{filename}")
async def delete_video(filename: str):
    try:
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
