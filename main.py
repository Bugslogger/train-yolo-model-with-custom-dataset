from convert_xml_to_yolo import convert_voc_to_yolo

# Example usage
xml_folder = 'test_dataset/annotations'  # Folder containing your .xml files
output_folder = 'trained_file'  # Folder to save YOLO .txt files
class_names = ['hard hat', 'helmet', 'hat','person','dog']  # List of your class names
images_path = 'test_dataset/images'

# below function call converts xml file into .txt file which will be used to train modal
convert_voc_to_yolo(xml_folder, output_folder, images_path, class_names)
