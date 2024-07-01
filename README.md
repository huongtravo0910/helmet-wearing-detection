# Helmet Detection using YOLOv10

This project demonstrates how to use a YOLOv10 model for helmet detection in images. The application is built using Streamlit to provide an interactive web interface.

## Prerequisites

- Python 3.7 or higher
- `pip` (Python package installer)


## Installation

1. **Clone the repository**:
    ```bash
    git clone git@github.com:huongtravo0910/helmet-wearing-detection.git
    ```

2. **Install the required packages**:
    ```bash
    pip install streamlit
    pip install cv2
    pip install numpy
    ```

3. **Check the pre-trained YOLOv10 model** 
 - You can either reuse the current pre-trained YOLOv10 model provided in the repository or:
 - Train a new model by following the instructions in `helmet_wearing_detection.ipynb`. Once trained, replace the `best.pt` in the repository with the new `best.pt` file found at `yolov10/runs/detect/train/weights/best.pt`.


## Running the Application

To start the Streamlit app, run the following command in your terminal:
```bash
streamlit run app.py
```

## Demo

https://github.com/huongtravo0910/helmet-wearing-detection/assets/66101016/37d7d179-c35f-435c-8e0d-d362876923c7

