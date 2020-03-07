# Object Detection and Tracking

A simple Python script to use a pretrained Darknet network and YOLOv3 to detect objects in a scene and track them across frames.

![Annotated Sample](./sample.gif)

## Usage

In `main.py`, update the path to a video or photo and enable the `gpu` attribute if you're running on GPU.

```
if __name__ == "__main__":
    # analyzer = Analyzer(video_path='./samples/chicago.mp4', gpu=False, save_output=True)
    analyzer = Analyzer(image_path='./samples/nyc.jpg', gpu=False, save_output=True)
    analyzer.visualize()
```

Run the script with

    python main.py

## Resources

This repo borrows heavily from [https://github.com/cfotache/pytorch_objectdetecttrack](https://github.com/cfotache/pytorch_objectdetecttrack
). Modifications were made to clean up the code and remove a couple of warnings from older libs.

### Object Tracking 

Object Tracking across scenes uses [SORT: A Simple, Online and Realtime Tracker](https://github.com/abewley/sort), copyright (C) 2016 Alex Bewley alex@dynamicdetection.com.



