# Object Detection and Tracking

Detect and track objects across frames using a pretrained Darknet network and YOLOv3 to detect objects in a scene and track them across frames. Part of my Twitch series on building computer vision components of self-driving machines.

![Annotated Sample](./sample.gif)

Video source: [https://www.videezy.com/free-video/traffic](https://www.videezy.com/free-video/traffic)

## Requirements

* Python 3
* [PyTorch](https://pytorch.org/get-started/locally/)
* Then `pip install -r requirements.txt`
* Finally, download weights for the network: `cd model && bash model/download_weights.sh`

## Usage

Run `main.py` with the following command line arguments,

```
usage: main.py [-h] [--video VIDEO] [--image IMAGE] [--gpu] [--save]

Annotate an image or video

optional arguments:
  -h, --help     show this help message and exit
  --video VIDEO  Path to MP4 video file
  --image IMAGE  path to image file
  --gpu          Run the network on a GPU
  --save         Save results
```

e.g.,

    python main.py --image samples/nyc.jpg --gpu --save

Press "q" to close the annotated window and exit the running program.

## Resources

This repo borrows heavily from [https://github.com/cfotache/pytorch_objectdetecttrack](https://github.com/cfotache/pytorch_objectdetecttrack
). Modifications were made to clean up the code and remove a couple of warnings from older libs.

### Object Tracking 

Object Tracking across scenes uses [SORT: A Simple, Online and Realtime Tracker](https://github.com/abewley/sort), copyright (C) 2016 Alex Bewley alex@dynamicdetection.com.



