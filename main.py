import numpy as np 
import pandas as pd 
import os

import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import cvzone
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2
import sys


def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))

    return transform


class AquariumDetection(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None):
        # the 3 transform parameters are reuqired for datasets.VisionDataset
        super().__init__(root, transforms, transform, target_transform)
        self.split = split #train, valid, test
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json")) # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
    
    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))
        
        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
        
        image = transformed['image']
        boxes = transformed['bboxes']
        
        new_boxes = [] # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        targ = {} # here is our transformed target
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        targ['image_id'] = torch.tensor([t['image_id'] for t in target])
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) 
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        return image.div(255), targ # scale images
    def __len__(self):
        return len(self.ids)

dataset_path ='./Dataset/'

#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes = len(categories.keys())
# print(categories)
# print('n_classes: ',n_classes)

classes = [i[1]['name'] for i in categories.items()]
# print(classes)

train_dataset = AquariumDetection(root=dataset_path, transforms=get_transforms(True))
# print(len(train_dataset))

def plot(sample,img_int):

    img = np.rollaxis(np.array(img_int), 0, 3)

    for i in range(len((sample[1]['boxes']).tolist())):

        ids=classes[sample[1]['labels'][i]]
        x1,y1,x2,y2=(sample[1]['boxes']).tolist()[i]
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h),l=20,t=3,colorR=(0,0,0))
        cvzone.putTextRect(img, f'{ids}', (x1-5, y1-10),
            scale=1, thickness=1,colorT=(255, 255, 255), colorR=(0, 0, 0))

    plt.imshow(img)
    plt.show()
    
# plot image
import random
num = random.randint(0,len(train_dataset))
sample = train_dataset[num]
img_int = torch.tensor(sample[0] * 255, dtype=torch.uint8)

plot(sample,img_int)

# lets load the faster rcnn model
model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn = collate_fn)

images,targets = next(iter(train_loader))
images = list(image for image in images)
targets = [{k:v for k, v in t.items()} for t in targets]
output = model(images, targets)

device = torch.device("cuda")
model = model.to(device)

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()
    
    
    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))

num_epochs=50

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)

torch.save(model, './weights/model_saved_50.pt')

############################################################

device = torch.device("cuda")
model = torch.load('./weights/model_saved_50.pt')
model.eval()
torch.cuda.empty_cache()


def inference(pred,img_int):

    im = np.rollaxis(np.array(img_int), 0, 3)

    for i in range(len(pred['boxes'][pred['scores'] > 0.8])):

        ids=classes[pred['labels'][pred['scores'] > 0.8].tolist()[i]]
        x1,y1,x2,y2=(pred['boxes'][pred['scores'] > 0.8]).tolist()[i]
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        # print(x1,y1,x2,y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(im, (x1, y1, w, h),l=20,t=3,colorR=(0,0,0))
        cvzone.putTextRect(im, f'{ids}', (x1-5, y1-10),
            scale=1, thickness=1,colorT=(255, 255, 255), colorR=(0, 0, 0))

    print(type(im))
    print(im.shape)
    plt.imshow(im)
    plt.show()


import random
num = random.randint(0,len(train_dataset))
img, _ = train_dataset[num]
img_int = torch.tensor(img*255, dtype=torch.uint8)
prediction = model([img.to(device)])
pred = prediction[0]

inference(pred,img_int)


