import os
import argparse

def separate_labels(input_dir, output_dir):
    person_dir = os.path.join(output_dir, 'person')
    ppe_dir = os.path.join(output_dir, 'ppe')
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(ppe_dir, exist_ok=True)

    classes_file = os.path.join(input_dir, 'classes.txt')
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    for label_file in os.listdir(input_dir):
        if label_file.endswith('.txt') and label_file != 'classes.txt':
            input_path = os.path.join(input_dir, label_file)
            person_output_path = os.path.join(person_dir, label_file)
            ppe_output_path = os.path.join(ppe_dir, label_file)

            with open(input_path, 'r') as infile, \
                 open(person_output_path, 'w') as person_outfile, \
                 open(ppe_output_path, 'w') as ppe_outfile:
                
                for line in infile:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    
                    if classes[class_id] == 'person':
                        person_outfile.write(line)
                    else:
                        new_class_id = class_id - 1
                        ppe_outfile.write(f"{new_class_id} {' '.join(parts[1:])}\n")

    print("Label separation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate person and PPE labels")
    parser.add_argument("input_dir", help="Path to the input directory containing YOLO format labels")
    parser.add_argument("output_dir", help="Path to the output directory for separated labels")
    args = parser.parse_args()

    separate_labels(args.input_dir, args.output_dir)