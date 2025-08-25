import torch
from pprint import pprint
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn , FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from argparse import ArgumentParser
import os
import numpy as np
import cv2

def get_args():
    parse = ArgumentParser(description="FasterRCNN Testing")
    
    parse.add_argument("--checkpoint",
                       type=str,
                       default="trained_model/best_rcnn.pt")
    
    parse.add_argument("--image_path",
                       type=str,
                       required=True)
    
    parse.add_argument("--threshold",
                       type=float,
                       default=0)
    
    args = parse.parse_args()

    return args

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = fasterrcnn_mobilenet_v3_large_fpn()
in_channels = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels , num_classes=21)
model = model.float()
model.to(device)

def image_test():
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    org_image = cv2.imread(args.image_path)
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image = np.transpose(image , (2 , 0 , 1))/255.
    image = [torch.from_numpy(image).to(device).float()]
        
    model.eval()

    with torch.no_grad():
        output = model(image)[0]
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        for box , label , score in zip(boxes , labels , scores):
            if score > args.threshold:
                xmin , ymin , xmax , ymax = box
                cv2.rectangle(org_image , (int(xmin),int(ymin)) , (int(xmax),int(ymax)) , (0,0,255) , 3)
                category = categories[label]
                cv2.putText(org_image , category , (int(xmin),int(ymin)) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,255,0) , 3 , cv2.LINE_AA)
                cv2.putText(org_image , str(round(score.item(),2)) , (int(xmin)+int(xmin),int(ymin)) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,255,0) , 3 , cv2.LINE_AA)
        cv2.imwrite("Prediction.jpg" , org_image)
        cv2.imshow("Prediction" , org_image)
        cv2.waitKey(0)

if __name__ == "__main__":
    args = get_args()
    image_test()