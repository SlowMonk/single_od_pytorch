import torchvision.transforms.functional as FT

from torch.utils.data import Dataset
from utils import region_shape_attributes_to_xywh, transform, resize
from torchvision import transforms
from albumentations.pytorch import ToTensor
import pandas as pd
import numpy as np
from PIL import Image
import glob
import json
import torch
import cv2
import os
import albumentations
from albumentations.pytorch.transforms import ToTensorV2,ToTensor
# this function will take the dataframe and vertically stack the image ids
# with no bounding boxes
from albumentations import (ToFloat,
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion,
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness,
    Flip, OneOf, Compose
)

from albumentations import (HorizontalFlip, ShiftScaleRotate, VerticalFlip, Normalize,Flip,
                            Compose, GaussNoise)
def process_bbox(df):
    df['bbox'] = df['region_shape_attributes'].apply(lambda x: eval(x))
    df['x'] = df['bbox'].apply(lambda x: x['x'])
    df['y'] = df['bbox'].apply(lambda x: x['y'])
    df['w'] = df['bbox'].apply(lambda x: x['width'])
    df['h'] = df['bbox'].apply(lambda x: x['height'])
    df['x'] = df['x'].astype(np.float)
    df['y'] = df['y'].astype(np.float)
    df['w'] = df['w'].astype(np.float)
    df['h'] = df['h'].astype(np.float)

    df.drop(columns=['bbox'], inplace=True)
    #     df.reset_index(drop=True)
    return df


class BalloonDataset(Dataset):
    def __init__(self, root, split,dataset_num,input_size):
        self.root = root
        self.split = split
        self.images = []
        self.split = split.upper()
        self.num = dataset_num
        if self.split == 'TRAIN':
            self.csv_path = '/home/jake/PycharmProjects/balloon_detection/ballon_datasets/via_region_data_{}_{}.csv'.format(self.split,self.num)
        else:
            self.csv_path = '/home/jake/PycharmProjects/balloon_detection/ballon_datasets/via_region_data_{}_70.csv'.format(self.split)
        self.df = None
        self.input_size= input_size


        rgb_mean = (0.4914, 0.4822, 0.4465)
        rgb_std = (0.2023, 0.1994, 0.2010)

        self.images = glob.glob(self.root +self.split +'_'+str(self.num)+'/*')

        # csv
        df = pd.read_csv(self.csv_path)
        self.df = process_bbox(df)


    def __getitem__(self, idx):
        image = self.images[idx]
        image_arr = cv2.imread(image, cv2.IMREAD_COLOR)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_arr /= 255.0


        image_id = str(image.split('.')[0])
        image_id = str(image_id.split('/')[-1]) + '.jpg'

        point = self.df[self.df['#filename'] == image_id]
        boxes = point[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = torch.ones((point.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((point.shape[0],), dtype=torch.int64)

        #transform resize image,boxes
        image_arr, boxes = transform(image_arr, boxes, self.split,self.input_size,labels=labels)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        target = {}
        target['boxes'] = boxes.float()
        target['labels'] = labels
        #target['image_id'] = objects['#filename']
        target['area'] = area.float()
        target['iscrowd'] = iscrowd.float()

        image_arr = FT.to_pil_image(image_arr)
        image_arr = np.array(image_arr)

        sample = {
            'image': FT.to_tensor(image_arr),
            'bboxes': target['boxes'],
            'labels': target['labels']
        }
        image = sample['image']
        target['boxes'] = torch.stack(tuple(map(torch.tensor,zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self):
        # print('__len__')
        return len(self.images)


def collate_fn(batch):
    return tuple(zip(*batch))

