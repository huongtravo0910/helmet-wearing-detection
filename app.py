import streamlit as st
from ultralytics import YOLOv10
import cv2
import numpy as np

TRAINED_MODEL_PATH = 'best.pt'
CONF_THRESHOLD = 0.2
IMG_SIZE = 640


def main():
    st.title('Helmet Detection')

    # Upload image
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting...")

        # Perform detection
        model = YOLOv10(TRAINED_MODEL_PATH)
        results = model.predict(image, imgsz=IMG_SIZE,
                                conf=CONF_THRESHOLD)

        # Convert results to a format we can use
        if isinstance(results, list):
            results = results[0]

        # Extract boxes, class names, and confidences
        boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        class_names = results.names  # Class names

        # Draw bounding boxes on the image
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            name = class_names[class_id]
            label = f"{name} {score:.2f}"
            color = (0, 255, 0) if name == 'helmet' else (
                255, 0, 0)  # Green for helmet, Red otherwise
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the processed image
        st.image(image, caption='Processed Image', use_column_width=True)


if __name__ == "__main__":
    main()
