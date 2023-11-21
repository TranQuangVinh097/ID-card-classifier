import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import random

aug_train = A.Compose(
            [
                A.Resize(224, 224),
                A.OneOf([
                    A.ChannelShuffle(p=0.35),
                    A.ToGray(p=0.35),
                    # A.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=5, num_shadows_upper=6, shadow_dimension=5, p = 0.15),
                    # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.75), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, 
                    #              num_flare_circles_upper=10, src_radius=150, src_color=(255, 255, 255), p=0.15)
                ], p=0.5),
                
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=0.3, brightness_by_max=True)
                ], p=0.5),
                A.OneOf([
                    A.Downscale (scale_min=0.5, scale_max=0.5, interpolation=None),
                    A.GaussianBlur(),
                    A.MotionBlur(blur_limit=[5,5])
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(150.0, 150.0)),
                    A.ISONoise(color_shift=(0.08, 0.08),intensity=(0.7, 0.7))
                ], p=0.5),
                # A.CoarseDropout(max_holes=10, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.4),
                ToTensorV2(),
            ]
        )

aug_val = A.Compose(
            [
                A.Resize(224, 224),
                ToTensorV2(),
            ]
        )

aug_mosaic = A.Compose(
            [
                A.Resize(224,224),
            ]
        )    

aug_perspect = A.Compose(
            [
                A.Resize(224,224),
                A.Perspective(scale=(0.05,0.2),fit_output=True,p = 1),
            ]
        )    

aug_rotate = A.Compose(
            [
                A.Rotate(limit=(-90,90),border_mode=cv2.BORDER_CONSTANT, p = 1),
                A.Resize(224,224),

            ]
        ) 

aug_cutout = A.Compose(
            [
                A.Resize(224,224),
                A.CoarseDropout (max_holes=20, max_height=32, max_width=32, min_holes=15, min_height=None, 
                         min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=1),
            ]
        )       

def offline_aug(dataset, index, cnt, img_path, aug_name, aug):
    img, label = dataset.__getitem__(index,'None')
    img = aug(image=img)["image"]
    label = label.tolist()
    
    file_name = aug_name + '/' + aug_name + '_' + str(cnt) + '.jpg'
    save_path = img_path + file_name
    cv2.imwrite(save_path, img)

    return file_name,label


def mosaic_aug(dataset,index,cnt, img_path):
    indices = [index] + [random.randint(0, len(dataset) - 1)  for _ in range(3)] 
    for i, index in enumerate(indices):
        # Load image
        img, label = dataset.__getitem__(index,'None')
        img = aug_mosaic(image=img)["image"]
        h = img.shape[0]
        w = img.shape[0]
        yc = h
        xc = w
        s = max(h, w)
        # place img in img4
        if i == 0:  # top left
            label4 = [0 for i in range(len(label))]
            img4 = np.full((s * 2, s * 2,img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles    
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)


        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b] # img4[ymin:ymax, xmin:xmax]
        
        x = label.tolist()

        for idx in range(len(x)):
            label4[idx] = (int)(label4[idx]) or (int)(x[idx])

    file_name = 'mosaic/mosaic_' + str(cnt) + '.jpg'
    save_path = img_path + file_name
    cv2.imwrite(save_path, img4)

    return file_name,label4


