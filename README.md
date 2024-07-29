# In-Store Analytics using Object Detection
This project implements Object Detection for performing In-Store Analytics using `yolov8n` as the per-trained model. This can help in getting using insights for the store manager to analyze and boost up yhe sales by monitoring the crowd at a particular instance in the store. To learn more about the project please refer this [article](link).

![Alt Text - description of the image](https://github.com/vansh-khaneja/In-Store-Analytics-using-Object-Detection/blob/main/pexels-shvetsa-3962285.jpg?raw=true)


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Execution](#execution)
- [Contact](#contact)

## Introduction

In this project, we used YOLOv8 model by [Ultralytics](https://docs.ultralytics.com/) for efficiently detecting customers from the video frames in real-time. This can help in getting useful data for the store manager to increase sales.

## Features

- Fast and efficient method
- Uses `yolov8n` model
- Real time support
- Optimizable to increase accuarcy

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/vansh-khaneja/In-Store-Analytics-using-Object-Detection
    cd In-Store-Analytics-using-Object-Detection
    ```

2. Set up the Python environment and install dependencies:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```


## Execution

1.Download the sample Video Dataset for this project [here](https://www.pexels.com/video/people-walking-inside-a-shopping-mall-4750076/) or you can try with your own dataset. Just change the path of the Video here.

```sh
    input_video_path = "c:/Users/testing.mp4"
```

2. You can also change the model based on the use case. Here in this project we have used ```yolov8n.pt``` model. Please refer [Ultraytics documentation](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) for learning about different model sizes. You can change the model here.
   
```sh
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

```

3.Execute the ```main.py``` file by running this command in terminal.

```sh
    python main.py
```


## Contact

For any questions or issues, feel free to open an issue on this repository or contact me at vanshkhaneja2004@gmail.com.

Happy coding!
