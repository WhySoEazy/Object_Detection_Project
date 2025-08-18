import torch
from torch.optim import SGD
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor , Compose , Resize , Normalize , RandomAffine , ColorJitter
from pprint import pprint
from VOC_Dataset import MyVOC
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn , FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from argparse import ArgumentParser
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchmetrics.detection import MeanAveragePrecision
torch.multiprocessing.set_sharing_strategy("file_system")

def get_args():
    parse = ArgumentParser(description="FasterRCNN Training")

    parse.add_argument("--data_path", 
                       type=str, 
                       default="C:/Users/triet/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1")
    
    parse.add_argument("--epochs",
                       type=int,
                       default=100)
    
    parse.add_argument("--batchs",
                       type=int,
                       default=8)
    
    parse.add_argument("--lr",
                       type=float,
                       default=1e-3)
    
    parse.add_argument("--momentum",
                       type=float,
                       default=0.9)
    
    parse.add_argument("--logging",
                       type=str,
                       default="tensorboard")
    
    parse.add_argument("--checkpoint",
                       type=str,
                       default=None)
    
    parse.add_argument("--trained_model",
                       type=str,
                       default="trained_model")

    parse.add_argument("--image_size",
                       type=int,
                       default=224)
    
    args = parse.parse_args()

    return args

def collate_fn(batch):
    images , labels = zip(*batch)
    return list(images) , list(labels)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    num_epoch = args.epochs
    batch_size = args.batchs 

    transform = Compose([
        ToTensor(),
        RandomAffine(
            degrees=(-5 , 5),
            translate=(0.05 , 0.05),
            scale=(0.85 , 1.15),
            shear=5
        ),
        ColorJitter(
            brightness=0.125,  
            contrast=0.5,   
            saturation=0.25,  
            hue=0.5
        )
    ])

    train_dataset = MyVOC(root=args.data_path , 
                    year="2012", 
                    image_set="train", 
                    download=False,
                    transform=transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_dataset = MyVOC(root=args.data_path , 
                    year="2012", 
                    image_set="val", 
                    download=False,
                    transform=ToTensor())


    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = fasterrcnn_mobilenet_v3_large_fpn(weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT ,
                                              trainable_backbone_layers = 0)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels , num_classes=len(train_dataset.categories))
    optimizer = SGD(params=model.parameters() , lr=args.lr , momentum=args.momentum)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_map = checkpoint["best_map"]
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        best_map = 0

    model.to(device)

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging , ignore_errors=True)

    if not os.path.isdir(args.trained_model):
            os.mkdir(args.trained_model)

    writer = SummaryWriter(args.logging)

    num_iters = len(train_dataloader)
    
    for epoch in range(start_epoch , start_epoch + num_epoch):
        #TrainingPhase
        model.train()
        progress_bar = tqdm(train_dataloader)
        train_loss = []
        for iter , (images , labels) in enumerate(progress_bar):

            #Forward
            images = [image.to(device) for image in images]
            labels = [{"boxes": target["boxes"].to(device) , "labels": target["labels"].to(device)} for target in labels]
            losses = model(images , labels)
            final_losses = sum([loss for loss in losses.values()])

            #Backward
            optimizer.zero_grad()
            final_losses.backward()
            optimizer.step()

            train_loss.append(final_losses.item())
            mean_train_loss = np.mean(train_loss)

            progress_bar.set_description("Epoch {}/{} , Loss {:0.4f}".format(epoch + 1 , num_epoch + start_epoch , mean_train_loss))

            writer.add_scalar("Train/Loss" , mean_train_loss , epoch * num_iters + iter)

        #ValidationPhase
        model.eval()
        progress_bar = tqdm(val_dataloader)
        metric = MeanAveragePrecision(iou_type='bbox')
        val_loss = []
        for iter , (images , labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                outputs = model(images)

            preds = []

            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].to("cpu"),
                    "scores": output["scores"].to("cpu"),
                    "labels": output["labels"].to("cpu")
                })

            targets = []

            for label in labels:
                targets.append({
                    "boxes": label["boxes"],
                    "labels": label["labels"]
                })

            metric.update(preds,targets)

        result = metric.compute()
        pprint(result)
        writer.add_scalar("Val/mAP" , result["map"] , epoch)
        writer.add_scalar("Val/mAP50" , result["map_50"] , epoch)
        writer.add_scalar("Val/mAP_75" , result["map_75"] , epoch)

        #SaveModel
        if result["map"] > best_map:
            checkpoint = {
                "best_map": best_map,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_rcnn.pt".format(args.trained_model))
            best_map = result["map"]

        checkpoint = {
            "epoch": epoch + 1,
            "best_map": best_map,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_rcnn.pt".format(args.trained_model))

        print("Best mAP: {}".format(best_map))

if __name__ == "__main__":
    args = get_args()
    train()