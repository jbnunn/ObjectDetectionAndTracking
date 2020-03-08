from config import Config
from sort import *
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
from model.darknet import *

import cv2
import os
import torch

class Analyzer():
    def __init__(self, video_path=None, image_path=None, gpu=True, save_output=False):
    
        self.img_size=416
        self.conf_thres=0.9
        self.nms_thres=0.4
        self.colors = Config.colors
        self.classes = Config.classes
        self.save_output = save_output
        
        self.model = Darknet(img_size=self.img_size)
        self.model.load_weights('model/yolov3.weights')

        if gpu:
            self.Tensor = torch.cuda.FloatTensor
            self.model.cuda()
        else:
            self.Tensor = torch.FloatTensor

        self.model.eval()

        if video_path:
            self.is_video = True
            self.video_path = video_path

            filename = os.path.splitext(video_path) 
            out_filename = filename[0] + "-annotated" + filename[1]
            self.out_filename = out_filename

        elif image_path:
            self.is_video = False
            self.image_path = image_path

            filename = os.path.splitext(image_path) 
            out_filename = filename[0] + "-annotated" + filename[1]
            self.out_filename = out_filename

    def detect_objects(self, img):

        # scale and pad image
        ratio = min(self.img_size/img.size[0], self.img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
            transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)), (128,128,128)),
            transforms.ToTensor(),
        ])
        
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(self.Tensor))
        
        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
        
        return detections[0]

    def visualize(self):
        if self.is_video:
            vid = cv2.VideoCapture(self.video_path)
            object_tracker = Sort() 

            cv2.namedWindow('Stream', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow('Stream', (600,400))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            ret, frame = vid.read()
            vw = frame.shape[1]
            vh = frame.shape[0]
            
            if self.save_output:
                outvideo = cv2.VideoWriter(self.out_filename,fourcc,20.0,(vw,vh))

            frames = 0
            start_time = time.time()

            while(True):
            
                ret, frame = vid.read()
                if not ret:
                    break
            
                frames += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(frame)
            
                detections = self.detect_objects(pilimg)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                img = np.array(pilimg)
                pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
                
                unpad_h = self.img_size - pad_y
                unpad_w = self.img_size - pad_x
            
                if detections is not None:
                    tracked_objects = object_tracker.update(detections.cpu())

                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                        color = self.colors[int(obj_id) % len(self.colors)]
                        cls = self.classes[int(cls_pred)]
                        cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                        cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                        cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                cv2.imshow('Stream', frame)
                
                if self.save_output:
                    outvideo.write(frame)

                ch = 0xFF & cv2.waitKey(1)
                if ch == 27 or ch == ord('q'):
                    break

            total_time = time.time()-start_time
            print(frames, "frames", total_time/frames, "s/frame")
            cv2.destroyAllWindows()
            
            if self.save_output:
                print(f"Saved {self.out_filename}")
                outvideo.release()

            sys.exit()

        elif not self.is_video:

            object_tracker = Sort() 

            cv2.namedWindow('Stream', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow('Stream', (600,400))

            while(True):

                frame = cv2.imread(self.image_path)
                vw = frame.shape[1]
                vh = frame.shape[0]
                
                start_time = time.time()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(frame)
                detections = self.detect_objects(pilimg)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                img = np.array(pilimg)
                pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
                
                unpad_h = self.img_size - pad_y
                unpad_w = self.img_size - pad_x
            
                if detections is not None:
                    tracked_objects = object_tracker.update(detections.cpu())

                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                        color = self.colors[int(obj_id) % len(self.colors)]
                        cls = self.classes[int(cls_pred)]
                        cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                        cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                        cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                cv2.imshow('Stream', frame)

                ch = 0xFF & cv2.waitKey(0)
                if ch == 27 or ch == ord('q'):
                    break

            total_time = time.time()-start_time
            print(f"Total_time: {total_time}s")
            cv2.destroyAllWindows()
            
            if self.save_output:
                print(f"Saved {self.out_filename}")
                cv2.imwrite(self.out_filename, frame)

            sys.exit()


if __name__ == "__main__":
    analyzer = Analyzer(video_path='./samples/chicago.mp4', gpu=False, save_output=True)
    #analyzer = Analyzer(image_path='./samples/nyc.jpg', gpu=False, save_output=True)
    analyzer.visualize()
