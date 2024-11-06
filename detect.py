
import cv2
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

def run_web_cam(model):
    # Execute inference with the YOLOv8s-world model on the specified image
    # webcam=cv2.VideoCapture(0) # uses your current systems webcam
    try: 
        rtsp_url = "rtsp://admin:Nissi$123@192.168.1.64/Streaming/Channels/101"  # Example RTSP URL
        webcam = cv2.VideoCapture(rtsp_url)
        
        if not webcam.isOpened():
            print("Error: Could nnot open webcam.")
            exit()

        frame_skip = 1
        frame_count = 0
        
        # Create a named window and set it to fullscreen
        cv2.namedWindow('Webcam - Object Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Webcam - Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while True:
            ret,frame=webcam.read()
            
            # Define class names (make sure they match the model's classes)
            classes = ["Helmet", "No-Helmet", "No-Vest", 
                "Person", "helmet", "no helmet", "no vest", 
                "no_helmet", "no_vest"]
            
            if not ret:
                print("Error: Could not read frame.")
                break
            
            frame_resized = cv2.resize(frame, (100,100))
            
            if frame_count % frame_skip == 0:
                results = model.predict(source=frame, device='cpu', conf=0.02, classes=[8,9,10,11,17,18,20])

                # Show results
                # results[0].show()

                # Process results
                for result in results:
                    # Draw bounding boxes and labels on the frame
                    boxes = result.boxes  # Get the bounding boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
                        conf = box.conf[0]  # Get confidence score
                        cls = int(box.cls[0])  # Get class id

                        # Draw bounding box and label
                        label = f"{model.names[cls]}: {conf:.2f}"  # Get class name and confidence
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw rectangle
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Draw label

            # Display the resulting frame with detections
            cv2.imshow('Webcam - Object Detection', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
        # When everything is done, release the capture
        webcam.release()
        cv2.destroyAllWindows()
    except  Exception as e:
        print("Error in Webcam Function: ", e)


def run_trained_model():
    # Load the ONNX model
    session = ort.InferenceSession('runs/detect/train/weights/best.onnx')
        
    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
        
    # Function to preprocess the image
    def preprocess_image(image):
        image_resized = cv2.resize(image, (640, 640))  # Resize to model input size
        image_normalized = image_resized.astype(np.float32) / 255.0  # Normalize
        input_data = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
        return input_data

    # Function to post-process the model outputs
    def postprocess_outputs(outputs, image_shape):
        boxes = outputs[0]  # Assuming the first output is the boxes
        scores = outputs[1]  # Assuming the second output is the scores
        class_ids = outputs[2]  # Assuming the third output is the class IDs
        print(f"postprocess_outputs:\n {boxes} \n {scores} \n {class_ids}")
        return boxes, scores, class_ids

    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_data = preprocess_image(frame)

        # Run inference
        outputs = session.run(None, {session.get_inputs()[0].name: input_data})

        # Post-process outputs
        boxes, scores, class_ids = postprocess_outputs(outputs, frame.shape)

        # Draw boxes on the frame
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Class {class_id}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame with detections
        cv2.imshow('Webcam Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

def rum_trained_pt_model():
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # model.set_classes(["Gloves",
    # "Goggles",
    # "Helmet",
    # "No-Helmet",
    # "No-Vest",
    # "Person",
    # "Reflective-Vest",
    # "Safety-Boot",
    # "Safety-Vest",
    # "Vest",
    # "Yelek",
    # "helmet",
    # "no helmet",
    # "no vest",
    # "no_helmet",
    # "no_vest",
    # "protective_suit",
    # "vest",
    # "worker"])
    
    run_web_cam(model)

# RUN 
# run_trained_model()
rum_trained_pt_model()