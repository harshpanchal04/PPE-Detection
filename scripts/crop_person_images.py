import os
import argparse
from PIL import Image

def calculate_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def crop_person_images(input_dir, input_dir2, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cropped_labels_dir = os.path.join(output_dir, 'cropped_labels')
    os.makedirs(cropped_labels_dir, exist_ok=True)

    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir2, 'yolo_annotations')

    image_count = 0
    person_count = 0

    for image_file in os.listdir(images_dir):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(images_dir, image_file)
            base_name = os.path.splitext(image_file)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            print(f"\nProcessing image: {image_file}")

            if not os.path.exists(label_path):
                print(f"  No label file found for {image_file}")
                continue

            image = Image.open(image_path)
            img_width, img_height = image.size

            with open(label_path, 'r') as label_file:
                labels = label_file.readlines()

            person_boxes = []
            ppe_labels = []

            for line in labels:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, width, height = map(float, parts)
                
                if class_id == 0:  # Assuming class_id 0 is 'person'
                    person_boxes.append((x_center, y_center, width, height))
                else:
                    ppe_labels.append((class_id, x_center, y_center, width, height))

            for i, (px_center, py_center, p_width, p_height) in enumerate(person_boxes):
                x_min = int((px_center - p_width/2) * img_width)
                y_min = int((py_center - p_height/2) * img_height)
                x_max = int((px_center + p_width/2) * img_width)
                y_max = int((py_center + p_height/2) * img_height)

                # Ensure bounding box is within image bounds
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(img_width, x_max), min(img_height, y_max)

                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                output_image_path = os.path.join(output_dir, f"{base_name}_person_{i+1}.jpg")
                cropped_image.save(output_image_path)

                # Process PPE labels for this person
                adjusted_ppe = []
                for ppe_class_id, ppe_x_center, ppe_y_center, ppe_width, ppe_height in ppe_labels:
                    # Convert to pixel coordinates
                    ppe_xmin = (ppe_x_center - ppe_width/2) * img_width
                    ppe_ymin = (ppe_y_center - ppe_height/2) * img_height
                    ppe_xmax = (ppe_x_center + ppe_width/2) * img_width
                    ppe_ymax = (ppe_y_center + ppe_height/2) * img_height

                    # Adjust coordinates relative to the cropped image
                    new_xmin = max(ppe_xmin - x_min, 0)
                    new_ymin = max(ppe_ymin - y_min, 0)
                    new_xmax = min(ppe_xmax - x_min, x_max - x_min)
                    new_ymax = min(ppe_ymax - y_min, y_max - y_min)

                    new_width = (new_xmax - new_xmin) / (x_max - x_min)
                    new_height = (new_ymax - new_ymin) / (y_max - y_min)
                    new_x_center = (new_xmin + new_xmax) / 2 / (x_max - x_min)
                    new_y_center = (new_ymin + new_ymax) / 2 / (y_max - y_min)

                    if new_width > 0 and new_height > 0:
                        distance = calculate_distance(new_x_center, new_y_center, 0.5, 0.5)
                        adjusted_ppe.append((distance, int(ppe_class_id), new_x_center, new_y_center, new_width, new_height))

                # Sort PPE by distance to the center of the person
                adjusted_ppe.sort(key=lambda x: x[0])

                # Write adjusted PPE labels
                output_label_path = os.path.join(cropped_labels_dir, f"{base_name}_person_{i+1}.txt")
                with open(output_label_path, 'w') as out_label:
                    for _, ppe_class_id, new_x_center, new_y_center, new_width, new_height in adjusted_ppe:
                        out_label.write(f"{ppe_class_id-1} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")

                print(f"  Saved cropped image: {output_image_path}")
                print(f"  Saved cropped label: {output_label_path}")
                person_count += 1

            image_count += 1

    print(f"\nProcessed {image_count} images")
    print(f"Cropped {person_count} person images")
    print("Person image cropping and label creation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop person images for PPE detection")
    parser.add_argument("input_dir", help="Path to the input directory containing images")
    parser.add_argument("input_dir2", help="Path to the input directory containing labels")
    parser.add_argument("output_dir", help="Path to the output directory for cropped images and labels")
    args = parser.parse_args()

    crop_person_images(args.input_dir, args.input_dir2, args.output_dir)