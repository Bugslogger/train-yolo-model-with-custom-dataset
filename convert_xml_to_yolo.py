import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_folder, output_folder, images_folder, class_names):
    print(xml_folder, output_folder, images_folder, class_names,"\n \n")
    
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        for xml_file in os.listdir(xml_folder):
            print("XML file: ",xml_file)
            if xml_file.endswith('.xml'):
                print("\n\nis XML file: True")
                # Parse the XML file
                tree = ET.parse(os.path.join(xml_folder, xml_file))
                root = tree.getroot()
                print("\n\nXML Parse: \n","tree: ",tree,"\n","root: ",root)

                # Get image filename from the XML (if available)
                image_filename = root.find('filename').text
                image_path = os.path.join(images_folder, image_filename)
                print("\n\nGetting image filename: ",image_filename)
                print("\nGetting image filename path: ",image_path,"\n\n")


                # Check if image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} does not exist.")

                # Get image dimensions
                width = int(root.find('size/width').text)
                height = int(root.find('size/height').text)
                print(f"\n\nImage Dimensions: \nWidth: {width} \nHeight: {height}")

                # Prepare the corresponding YOLO .txt file
                yolo_file = os.path.join(output_folder, xml_file.replace('.xml', '.txt'))
                print("\n\nis file Replaced: ", yolo_file)

                with open(yolo_file, 'w') as f:
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text
                        class_id = class_names.index(class_name)  # Get the class ID
                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)

                        # Convert to YOLO format (center_x, center_y, width, height)
                        center_x = (xmin + xmax) / (2 * width)
                        center_y = (ymin + ymax) / (2 * height)
                        bbox_width = (xmax - xmin) / width
                        bbox_height = (ymax - ymin) / height

                        # Write to the YOLO file
                        f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")
            else:
                print("is XML file: False")
    except Exception as e:
        print(e)
        
    
                    
#  Here this code converts XML file to yolo model extension file
#  as in the dataset we have annotation in .xml
#  So, first we need to convert our .xml to yolo for the  model to work properly
