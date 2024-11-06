import cv2
import os

def draw_labels(image_path, annotation_path, class_names):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Read the annotations
    with open(annotation_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Skipping malformed line: {line}")
                continue  # Skip this line if it doesn't have exactly 5 values
            class_id, center_x, center_y, box_width, box_height = map(float, parts)

            # Convert from normalized to pixel values
            x1 = int((center_x - box_width / 2) * width)
            y1 = int((center_y - box_height / 2) * height)
            x2 = int((center_x + box_width / 2) * width)
            y2 = int((center_y + box_height / 2) * height)

            # Draw the rectangle and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{class_names[int(class_id)]}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

def main(images_dir, annotations_dir, output_dir, class_names):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all images in the directory
    for image_file in os.listdir(images_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(images_dir, image_file)
            annotation_path = os.path.join(annotations_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

            if os.path.exists(annotation_path):
                labeled_image = draw_labels(image_path, annotation_path, class_names)
                output_path = os.path.join(output_dir, image_file)
                cv2.imwrite(output_path, labeled_image)
                print(f"Labeled image saved to: {output_path}")
            else:
                print(f"Annotation file does not exist for: {image_file}")

if __name__ == "__main__":
    # Define directories and class names
    images_dir = 'datasets/valid/images'  # Path to the directory containing images
    annotations_dir = 'datasets/valid/labels'  # Path to the directory containing .txt annotation files
    output_dir = 'yolo_test_dataset/images/val'  # Path to save labeled images

    # List of class names (make sure to match the order with class IDs in your annotations)
    class_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "Glass",
    "Gloves",
    "Goggles",
    "Helmet",
    "No-Helmet",
    "No-Vest",
    "Person",
    "Reflective-Vest",
    "Safety-Boot",
    "Safety-Vest",
    "Vest",
    "Yelek",
    "helmet",
    "no helmet",
    "no vest",
    "no_helmet",
    "no_vest",
    "protective_suit",
    "safety vest - v1 2023-11-19 5-12pm",
    "vest",
    "worker",
  ]
  # Add your class names here

    main(images_dir, annotations_dir, output_dir, class_names)