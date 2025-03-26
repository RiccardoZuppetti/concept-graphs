import os
import time
import random
import torch
from ultralytics import YOLO, SAM
import numpy as np
import cv2

from conceptgraph.utils.vlm import get_obj_captions_from_image_ollama, get_obj_rel_from_image_ollama

# Load YOLO and SAM models
# detection_model = YOLO("yolov8l.pt")  # YOLO LARGE
detection_model = YOLO('yolov8l-world.pt') # YOLOv8
# sam_predictor = SAM("sam_l.pt")  # SAM
#sam_predictor = SAM("mobile_sam.pt")  # MOBILE SAM
sam_predictor = SAM("sam2_l.pt")  # SAM 2

# Folder to save annotated images
# ANNOTATED_IMAGES_DIR = "annotated_images"
# os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)  # Create folder if it doesn't exist

# def visualize_and_save(image_path, final_detections, xyxy_np):
#     """
#     Draws bounding boxes and labels on an image and saves it.

#     Args:
#         image_path (str): Path to the original image.
#         final_detections (list): List of detected objects in the format ["1: car", "2: chair"].
#         xyxy_np (np.array): Bounding boxes of detected objects.
#     """
#     # Load the image
#     image = cv2.imread(image_path)

#     for idx, detection in enumerate(final_detections):
#         obj_id, obj_name = detection.split(": ")  # Extract index & object name

#         # Get bounding box
#         if idx < len(xyxy_np):  # Ensure we don't exceed available boxes
#             x1, y1, x2, y2 = xyxy_np[idx]

#             # Draw bounding box
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box

#             # Put text label
#             cv2.putText(image, f"{obj_name}", (int(x1), int(y1) - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Save the annotated image
#     save_path = os.path.join(ANNOTATED_IMAGES_DIR, os.path.basename(image_path))
#     cv2.imwrite(save_path, image)
#     print(f"Annotated image saved: {save_path}")

def extract_label_list(image_path, class_txt_path, min_mask_area=1000):
    """
    Extracts a formatted list of detections in the format: ["1: object1", "2: object2", ...]
    using YOLO for object detection and MobileSAM for segmentation.

    Args:
        image_path (str): Path to the image file.
        class_txt_path (str): Path to the .txt file containing object class names.
        min_mask_area (int): Minimum number of pixels for a valid mask.

    Returns:
        list: A list of strings formatted as ["1: object1", "2: object2", ...] (filtered using SAM).
    """

    bg_classes = ["wall", "floor", "ceiling"]
    classes = [line.strip() for line in open(class_txt_path)]
    classes = [cls for cls in classes if cls not in bg_classes] # remove background classes
    detection_model.set_classes(classes)

    # Run YOLO detection
    results = detection_model.predict(image_path, conf=0.1, verbose=False)
    
    # Extract class IDs and bounding boxes
    detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
    xyxy_tensor = results[0].boxes.xyxy
    xyxy_np = xyxy_tensor.cpu().numpy()

    # Map detected class IDs to class names
    detection_class_labels = [
        classes[class_id] if class_id < len(classes) else f"unknown_{class_id}"
        for class_id in detection_class_ids
    ]

    # Check if any detections were made
    if xyxy_tensor.numel() == 0:
        return []
    
    # Run SAM segmentation on detected bounding boxes
    sam_out = sam_predictor.predict(image_path, bboxes=xyxy_tensor, verbose=False)
    masks_tensor = sam_out[0].masks.data
    masks_np = masks_tensor.cpu().numpy()

    valid_detections = []
    for idx, mask in enumerate(masks_np):
        # Check if the mask contains enough pixels to be a valid object
        if np.sum(mask) > min_mask_area:  # Minimum mask size (adjustable)
            valid_detections.append(f"{len(valid_detections) + 1}: {detection_class_labels[idx]}")
    
    # visualize_and_save(image_path, valid_detections, xyxy_np)

    return valid_detections  # Return only valid detections based on SAM masks


def process_images_in_folder(folder_path, class_txt_path, num_images=100):
    """
    Processes a folder of images and extracts detections using YOLO + SAM.

    Args:
        folder_path (str): Path to the image folder.
        class_txt_path (str): Path to the .txt file containing object class names.
        num_images (int): Number of images to process (randomly selected).

    Returns:
        dict: A dictionary with image names as keys and label lists as values.
    """
    # Get all images in the folder
    all_images = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".jpg")
    ]

    # Select 100 random images (or fewer if not enough available)
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    results = {}
    total_time = 0.0

    # Process each image
    for image_path in selected_images:
        print(f"\nProcessing image: {image_path}")
        
        start_time = time.time()
        label_list = extract_label_list(image_path, class_txt_path)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        print(f"Detected objects: {label_list} | Time taken: {elapsed_time:.2f}s")

        results[image_path] = label_list

    # Summary statistics
    avg_time = total_time / len(selected_images) if selected_images else 0
    print("\n================== Detection Benchmark Report ==================")
    print(f"Total images processed: {len(selected_images)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing time per image: {avg_time:.2f} seconds")
    print("===============================================================")

    return results


if __name__ == "__main__":
    folder_path = "/home/ubuntu/Downloads/color_images"
    class_txt_path = "/home/ubuntu/my_code/concept-graphs/conceptgraph/scannet200_classes.txt"

    results = process_images_in_folder(folder_path, class_txt_path, num_images=100)

    total_images = 100
    total_time = 0.0
    success_count = 0
    detailed_metrics = []

    for img, labels in results.items():
        print(f"\nImage: {img}")
        print("Detections:", labels)
        print()

        start_time_vlm = time.time()
        print("VLM Outputs:")
        print("============")
        result_vlm_rel = get_obj_rel_from_image_ollama(img, labels)
        result_vlm_cap = get_obj_captions_from_image_ollama(img, labels)
        elapsed = time.time() - start_time_vlm
        print("============")

        total_time += elapsed

        # Consider the result valid if it is not an empty list
        if result_vlm_rel and result_vlm_cap:
            success_count += 1

        detailed_metrics.append({
            "image": img,
            "result_rel": result_vlm_rel,
            "result_cap": result_vlm_cap,
            "elapsed_time_sec": round(elapsed, 2)
        })

    avg_time = total_time / total_images if total_images else 0
    success_rate = (success_count / total_images) * 100 if total_images else 0

    # Final metrics report VLM
    print("\n================== VLM Metrics Report ==================")
    print(f"Images processed: {total_images}")
    print(f"Valid responses: {success_count} ({success_rate:.2f}%)")
    print(f"Average processing time per image: {avg_time:.2f} seconds")
    # print("Details for each image:")
    # for metric in detailed_metrics:
    #     print(f"  - {metric['image']}: Time = {metric['elapsed_time_sec']} sec, Relations = {metric['result_rel']}, Captions = {metric['result_cap']}")
    print("========================================================")