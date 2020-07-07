
import torchvision.transforms.functional as FT
import xml.etree.ElementTree as ET
#from engine import train_one_epoch, evaluate
import torch
import random
from torchvision import *
import json
import os
import numpy as np
from collections import defaultdict
import numpy as np
import torch
from torchvision.ops.boxes import box_iou
import cv2
from datasets import *
import torchvision
import matplotlib.pyplot as plt
import os
import time
import shutil
from utils import *
from datasets import *
import albumentations
from albumentations import *
from matplotlib import patches

def region_shape_attributes_to_xywh(arr):
    # {'name': 'rect', 'x': 849, 'y': 620, 'width': 332, 'height': 416},
    return [arr['x'], arr['y'], arr['width'], arr['height']]

def get_bboxes(bboxes, col, bbox_format = 'pascal_voc', color='white'):
    for i in range(len(bboxes)):
        x_min = bboxes[i][0]
        y_min = bboxes[i][1]
        x_max = bboxes[i][2]
        y_max = bboxes[i][3]
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min),
                                 width, height,
                                 linewidth=2,
                                 edgecolor=color,
                                 facecolor='none')
        col.add_patch(rect)
def transform(image, boxes,split,input_size,labels):

    assert split in {'TRAIN', 'TEST'}

    # print('transform_before_boxes->',boxes)
    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ig, ax = plt.subplots(1, 2, figsize=(24, 24))
    #ax = ax.flatten()

    new_image = image
    new_boxes = boxes
    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        if random.random() <0.5:
            new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        new_image = torch.as_tensor(new_image)
        boxes = torch.tensor(boxes)
        # 0.5 distort or brightness of the image
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        #if random.random() < 0.5:
        #    new_image, new_boxes = flip(new_image, new_boxes)

        #HorizontalFlip
        #if random.random() < 0.5:
        try:
                #print('aug_result->',aug_result)
            if random.random() < 0.3:
                if random.random() < 0.3:
                    augment = albumentations.HorizontalFlip(p=1)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                if random.random() < 0.3:
                    augment = albumentations.HorizontalFlip(p=0.8)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                if random.random() < 0.3:
                    augment = albumentations.HorizontalFlip(p=0.2)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                # print('aug_result->',aug_result)
            if random.random() < 0.3:
                if random.random() < 0.3:
                    augment = albumentations.ToGray(p=1)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)
                if random.random() < 0.3:
                    augment = albumentations.ToGray(p=0.6)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)
                if random.random() < 0.3:
                    augment = albumentations.ToGray(p=0.9)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)
            if random.random() < 0.3:
                if random.random() < 0.3:
                    augment = albumentations.RandomBrightnessContrast(p=1)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)
                if random.random() < 0.3:
                    augment = albumentations.RandomBrightnessContrast(p=0.2)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)
                if random.random() < 0.3:
                    augment = albumentations.RandomBrightnessContrast(p=0.7)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)
            if random.random() < 0.3:
                if random.random() < 0.3:
                    augment = albumentations.VerticalFlip(p=2)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)
                if random.random() < 0.3:
                    augment = albumentations.VerticalFlip(p=0.6)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)
                if random.random() < 0.3:
                    augment = albumentations.VerticalFlip(p=0.2)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)
                if random.random() < 0.3:
                    augment = albumentations.VerticalFlip(p=0.9)
                    aug = albumentations.Compose([augment],
                                                 bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
                    aug_result = aug(image=np.array(new_image), bboxes=new_boxes, labels=labels)
                    new_image = aug_result['image']
                    new_boxes = aug_result['bboxes']
                    new_image = FT.to_pil_image(new_image)
                    # print('aug_result->',aug_result)

        except:
            pass

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes,dims=(input_size, input_size))
    # print('transform_after_boxes->', new_boxes)

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    #new_image = FT.normalize(new_image, mean=mean, std=std)
    return new_image, new_boxes

def resize(image, boxes, dims=(300, 300), return_percent_coords=True):

    # Resize image

    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_dims =  torch.FloatTensor([dims[0], dims[0], dims[0], dims[0]]).unsqueeze(0)
    boxes = torch.as_tensor(boxes)
    old_dims = torch.as_tensor(old_dims)
    new_boxes = boxes * (new_dims/old_dims)

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims
    return new_image, new_boxes

# TODO later
def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    #print('photometric_distort')
    image = FT.to_tensor(image)
    image=FT.to_pil_image(image)
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image

# TODO later
def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes

# TODO later
def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)
    #print('new_image->',new_image)
    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def calculate_precision(boxes_true: torch.tensor, boxes_pred: torch.tensor, confidences: list, threshold=0.5) -> float:
    """Calculates precision for GT - prediction pairs at one threshold."""

    confidences = np.array(confidences)

    # edge case for no ground truth boxes
    if boxes_true.size(1) == 0:
        return 0.

    iou = box_iou(boxes1=boxes_pred, boxes2=boxes_true)

    pr_matches = set()
    gt_matches = set()

    # for each ground truth box, get list of pred boxes it matches with
    match_candidates = (iou >= threshold).nonzero()
    GT_PR_matches = defaultdict(list)
    for PR, GT in match_candidates:
        GT_PR_matches[GT.item()].append(PR.item())

    # Find which pred matches a GT box
    for GT, PRs in GT_PR_matches.items():
        # if multiple preds match a single ground truth box,
        # select the pred with the highest confidence
        if len(PRs) > 1:
            pr_match = PRs[confidences[PRs].argsort()[-1]]
        # else only a single pred matches this GT box
        else:
            pr_match = PRs[0]

        # only if we haven't seen a pred before can we mark a PR-GT pair as TP
        # otherwise the pred matches a different GT box and this GT might instead be a FN
        if pr_match not in pr_matches:
            gt_matches.add(GT)

        pr_matches.add(pr_match)

    TP = len(pr_matches)

    pr_idx = range(iou.size(0))
    gt_idx = range(iou.size(1))

    FP = len(set(pr_idx).difference(pr_matches))
    FN = len(set(gt_idx).difference(gt_matches))

    return TP / (TP + FP + FN)


def calculate_mean_precision(boxes_true: torch.tensor, boxes_pred: torch.tensor, confidences: np.array,
                             thresholds=(0.5,)):
    """Calculates average precision over a set of thresholds"""

    precision = np.zeros(len(thresholds))

    for i, threshold in enumerate(thresholds):
        precision[i] = calculate_precision(boxes_true=boxes_true, boxes_pred=boxes_pred, confidences=confidences,
                                           threshold=threshold)
    return precision.mean()

def test_calc_precision():
    boxes_true = torch.tensor([
        [0., 0., 10., 10.],     # GT1
        [0., 0., 12., 10.]      # GT2
    ])
    boxes_pred = torch.tensor([
        [0., 0., 0., 0.],      # P1
        [0., 0., 0., 0.],       # P2
        [0., 0., 10., 10.]
    ])
    confidences = [.5, .9]
    score = calculate_precision(boxes_true=boxes_true, boxes_pred=boxes_pred, confidences=confidences, threshold=.5)
    #assert score == 1.

    confidences = [.9, .5]
    score = calculate_precision(boxes_true=boxes_true, boxes_pred=boxes_pred, confidences=confidences, threshold=.5)
    #assert score == 1/3

    score = calculate_precision(boxes_true=boxes_true, boxes_pred=boxes_pred,
                                confidences=confidences, threshold=.5)
    print('test_calc_precision->',score)
    #assert score == 0


def make_true_boxes(df,image_num):
    x = df[df['#filename'] == df['#filename'][image_num]]['x'].values
    y = df[df['#filename'] == df['#filename'][image_num]]['y'].values
    w = df[df['#filename'] == df['#filename'][image_num]]['w'].values
    h = df[df['#filename'] == df['#filename'][image_num]]['h'].values

    true_boxes = []
    for i in range(len(x)):
        true_boxes.append([x[i], y[i], x[i] + w[i], y[i] + h[i]])
    return torch.tensor(true_boxes)


def make_true_boxes_new_scale(df, image_num,input_size):
    #print('make_true_boxes_new_scale[B]')
    x = df[df['#filename'] == df['#filename'][image_num]]['x'].values
    y = df[df['#filename'] == df['#filename'][image_num]]['y'].values
    w = df[df['#filename'] == df['#filename'][image_num]]['w'].values
    h = df[df['#filename'] == df['#filename'][image_num]]['h'].values

    # print(x,y,w,h)
    true_boxes = []
    for i in range(len(x)):
        new_x = x[i] * (input_size / w[i])
        new_y = y[i] * (input_size / h[i])
        new_w = input_size
        new_h = input_size
        true_boxes.append([(x[i] * (input_size / w[i])),
                           y[i] * (input_size / h[i]),
                           (x[i] * (input_size / w[i])) + (input_size),
                           (y[i] * (input_size / h[i])) + (input_size)]
                          )

    true_boxes
    return torch.tensor(true_boxes)
def make_input_image(img_path,input_size):
    image_arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_arr /= 255.0
    new_image = image_arr

    new_image = FT.to_tensor(new_image)
    new_image = FT.to_pil_image(new_image)
    new_image = FT.resize(new_image, (input_size, input_size))
    new_image = FT.to_tensor(new_image)
    new_image = FT.to_pil_image(new_image)
    new_image_model = FT.to_tensor(np.array(new_image)).cuda()
    return new_image_model
