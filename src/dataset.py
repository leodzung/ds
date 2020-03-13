import pandas as pd
import albumentations
import joblib
import numpy as np
import torch
import cv2

from PIL import Image

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, img_height, img_width, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back
    X_THREADHOLD = 17
    Y_THREADHOLD = 10
    xmin = xmin - X_THREADHOLD if (xmin > X_THREADHOLD) else 0
    ymin = ymin - Y_THREADHOLD if (ymin > Y_THREADHOLD) else 0
    xmax = xmax + X_THREADHOLD if (xmax < img_width - X_THREADHOLD) else img_width
    ymax = ymax + Y_THREADHOLD if (ymax < img_height - Y_THREADHOLD) else img_height
    img = img0[ymin:ymax,xmin:xmax]
    
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(img_height, img_width))

class BengaliDatasetTrain:
    def __init__(self, fold, img_height, img_width, mean, std, train=True, crop=True):
        df = pd.read_csv("../input/train_v2.csv")
        df = df[["image_id", "grapheme_root", "vowel_diacritic", "consonant_diacritic", "fold", "unseen"]]

        if train:
            train_idx = np.where((df['fold'] != fold) & (df['unseen'] == 0))[0]
            df = df.loc[train_idx].reset_index(drop=True)
        else:
            valid_idx = np.where((df['fold'] == fold) | (df['unseen'] != 0))[0]
            df = df.loc[valid_idx].reset_index(drop=True)
        
        self.img_height = img_height
        self.img_width = img_width

        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        self.crop = crop

        if not train:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),

                albumentations.OneOf([
                    albumentations.ShiftScaleRotate(scale_limit=.15, rotate_limit=20, 
                                                    border_mode=cv2.BORDER_CONSTANT),
                    albumentations.IAAAffine(shear=20, mode='constant'),
                    albumentations.IAAPerspective(),
                    albumentations.Cutout(num_holes=8, 
                                          max_h_size=img_height // 8, 
                                          max_w_size=img_width // 8),
                ]),

                albumentations.Normalize(mean, std, always_apply=True)
            ])

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, item):
        image = joblib.load(f"../input/image_pickles/{self.image_ids[item]}.pkl")
        image = image.reshape(137, 236).astype(float)
        if self.crop:
            image = crop_resize(image, self.img_height, self.img_width)
        image = Image.fromarray(image).convert("RGB")
        image = self.aug(image=np.array(image))["image"]
        cv2.imwrite(f"../input/images/{self.image_ids[item]}.png", image)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "grapheme_root": torch.tensor(self.grapheme_root[item], dtype=torch.long),
            "vowel_diacritic": torch.tensor(self.vowel_diacritic[item], dtype=torch.long),
            "consonant_diacritic": torch.tensor(self.consonant_diacritic[item], dtype=torch.long)
        }