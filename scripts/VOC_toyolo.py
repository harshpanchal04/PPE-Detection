import os
import argparse
import xml.etree.ElementTree as ET

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(input_dir, output_dir, image_id):
    in_file = os.path.join(input_dir, "labels", f"{image_id}.xml")
    out_file = os.path.join(output_dir, f"{image_id}.txt")
    
    if not os.path.exists(in_file):
        print(f"Warning: Input file not found: {in_file}")
        return

    try:
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        with open(out_file, "w") as f:
            for obj in root.iter("object"):
                cls = obj.find("name").text
                if cls not in classes:
                    print(f"Warning: Class '{cls}' not found in classes.txt. Skipping object in {image_id}")
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find("bndbox")
                b = (float(xmlbox.find("xmin").text), float(xmlbox.find("ymin").text),
                     float(xmlbox.find("xmax").text), float(xmlbox.find("ymax").text))
                bb = convert_coordinates((w, h), b)
                f.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
        print(f"Converted: {image_id}")
    except ET.ParseError:
        print(f"Error: Unable to parse XML file: {in_file}")
    except Exception as e:
        print(f"Error processing {image_id}: {str(e)}")

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    labels_dir = os.path.join(input_dir, "labels")
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        return

    for filename in os.listdir(labels_dir):
        if filename.endswith(".xml"):
            image_id = os.path.splitext(filename)[0]
            convert_annotation(input_dir, output_dir, image_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="datasets", help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, default="yolo_annotations", help="Path to the output directory")
    args = parser.parse_args()
    
    classes_file = os.path.join(args.input_dir, "classes.txt")
    if not os.path.exists(classes_file):
        print(f"Error: classes.txt not found in {args.input_dir}")
        exit(1)

    with open(classes_file, "r") as f:
        classes = f.read().splitlines()
    
    main(args)