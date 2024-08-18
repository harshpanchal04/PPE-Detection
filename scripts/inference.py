import argparse
import cv2
import os
import torch
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def draw_boxes(image, boxes, labels, confidences):
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Green color for all boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_image(image_path, ppe_model, output_dir):
    try:
        logger.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # PPE detection on the full image
        ppe_results = ppe_model(image)[0]
        ppe_boxes = ppe_results.boxes.xyxy.cpu().numpy()
        ppe_labels = [ppe_model.names[int(cls)] for cls in ppe_results.boxes.cls.cpu().numpy()]
        ppe_confs = ppe_results.boxes.conf.cpu().numpy()
        
        # Draw PPE boxes
        draw_boxes(image, ppe_boxes, ppe_labels, ppe_confs)
        
        # Save the output image
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        logger.info(f"Saved processed image: {output_path}")
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Perform inference using PPE detection model')
    parser.add_argument('input_dir', help='Path to the input directory containing images')
    parser.add_argument('output_dir', help='Path to the output directory for annotated images')
    parser.add_argument('person_det_model', help='Path to the person detection model weights (not used)')
    parser.add_argument('ppe_detection_model', help='Path to the PPE detection model weights')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    ppe_model = YOLO(args.ppe_detection_model).to(device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)

    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(args.input_dir, image_file)
        logger.info(f"Processing image {i}/{total_images}: {image_file}")
        process_image(image_path, ppe_model, args.output_dir)

    logger.info(f"Inference completed. Annotated images saved in {args.output_dir}")

if __name__ == '__main__':
    main()