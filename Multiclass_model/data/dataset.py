import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import random 
from data.aug_option import *
import copy
from data.prefetch import PrefetchLoader

Label_lbl = ['Banh_chung','Cay_dao','Cay_mai']
class MyDataset(Dataset):
    """
    Dataset for inference
    """

    def __init__(self, image_dir, image,augument):
        self.image_dir = image_dir
        self.images = image['PATH'].tolist()
        self.labels = image[['Banh_chung','Cay_dao','Cay_mai']].values.tolist()
        self.aug = augument


    def __add__(self, images, labels): 
        self.images.append(images)
        self.labels.append(labels)

    def __len__(self):
        return len(self.images)
    def __getlabel__(self):
        return self.labels
    def __classlen__(self):
        label_list = np.array(self.labels)
        label_list = np.transpose(label_list)
        class_len = [sum(label_list[i]) for i in range(len(label_list))]
        return class_len
    def __adddata__(self,dataset):
        self.images.extend(dataset.images)
        self.labels.extend(dataset.labels)

    def __getitem__(self,idx,mode='train'):
        img_path = self.image_dir + self.images[idx]
        img_label = torch.tensor(self.labels[idx],dtype=torch.float16)

        image = cv2.imread(img_path)
        if image is None:
            raise Exception(f"Image not found. Directory: {img_path}")
            
        if (mode == 'train'):
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.aug(image=image)["image"]
        return (image, img_label)
    
def add_offline_data(
    dataset: MyDataset,
    base_dataset: MyDataset,
    num_img: int,
    start: int,
    mode: str,
    img_path: str 
):
    
    for i in range(num_img):
        idx = random.randint(0, len(base_dataset) - 1)
        if (mode == 'mosaic'): 
            path, label = mosaic_aug(base_dataset, idx, i + start, img_path)
        elif (mode == 'perspect'):
            path, label = offline_aug(base_dataset, idx, i + start, img_path, 'perspect', aug_perspect)
        elif (mode == 'cutout'):
            path, label = offline_aug(base_dataset, idx, i + start, img_path, 'cutout', aug_cutout)
        elif (mode == 'rotate'):
            path, label = offline_aug(base_dataset, idx, i + start, img_path, 'rotate', aug_rotate)

        dataset.__add__(path,label)

    return dataset

def gen_offline_data(
        train_dataset: MyDataset, 
        val_dataset: MyDataset, 
        config = None
    ):

    img_path = config['DIRECTORY']['DATA_PATH']
    offline_augment = config['TRAIN_OPTION']['OFFLINE_AUG']
    # 'perspect', 'mosaic',,'rotate','cutout','cutmix'
    base_train_dataset = copy.deepcopy(train_dataset)
    base_val_dataset = copy.deepcopy(val_dataset)
    for aug in offline_augment:
        add_offline_data(train_dataset,base_train_dataset,100,0,aug,img_path)
        add_offline_data(val_dataset,base_val_dataset, 100,101,aug,img_path)

def update_dataframe(
    temp_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame 
):
    
    num_of_train_data = len(temp_df)*75//100
    train_df = pd.concat([train_df,temp_df.iloc[0:num_of_train_data].reset_index(drop=True)], ignore_index = True)
    val_df = pd.concat([val_df,temp_df[num_of_train_data:].reset_index(drop=True)], ignore_index = True)
    
    return train_df,val_df

def label_summary(
    train_dataset: MyDataset,
    val_dataset: MyDataset
):
    
    lbls_train = train_dataset.__getlabel__()
    lbls_val = val_dataset.__getlabel__()
    lbls_train.extend(lbls_val)
    banh_chung_cnt = 0
    cay_dao_cnt = 0
    cay_mai_cnt = 0
    print(len(lbls_train))
    for i in range(len(lbls_train)):
        banh_chung_cnt += lbls_train[i][0]
        cay_dao_cnt += lbls_train[i][1]
        cay_mai_cnt += lbls_train[i][2]
        
    print(f'total banh chung sample: {banh_chung_cnt}')
    print(f'total cay dao sample: {cay_dao_cnt}')
    print(f'total cay mai sample: {cay_mai_cnt}')
    
    
    banh_chung_cnt = 0
    cay_dao_cnt = 0
    cay_mai_cnt = 0
    for i in range(len(lbls_train)):
        banh_chung_cnt += (int)(lbls_train[i][0] == 0)
        cay_dao_cnt += (int)(lbls_train[i][1] == 0)
        cay_mai_cnt += (int)(lbls_train[i][2] == 0)
        
    print(f'total not banh chung sample: {banh_chung_cnt}')
    print(f'total not cay dao sample: {cay_dao_cnt}')
    print(f'total not cay mai sample: {cay_mai_cnt}')
    
def data_loader(config):
    
    data_dir = config['DIRECTORY']['DATA_PATH']
    csv_path = config['DIRECTORY']['DATA_CSV']
    df_columns = ['PATH'] + config['LABEL_OPTION']['LABEL']
    batch_size = config["TRAIN_OPTION"]["BATCH_SIZE"]

    train_df = pd.DataFrame(columns = df_columns)
    val_df = pd.DataFrame(columns = df_columns)

    df = pd.read_csv(csv_path)
    for col in df.columns:
        if (col != 'PATH'):
            temp_df = df[df[col] != 0]
            train_df,val_df = update_dataframe(temp_df,train_df,val_df)
    temp_df = df.loc[df['Banh_chung'] + df['Cay_dao'] + df['Cay_mai'] == 0]
    train_df,val_df = update_dataframe(temp_df,train_df,val_df)
   
   
    train_dataset = MyDataset(data_dir, train_df, aug_train)
    val_dataset = MyDataset(data_dir, val_df, aug_val)

    gen_offline_data(train_dataset,val_dataset,config)
    label_summary(train_dataset,val_dataset)


    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    return PrefetchLoader(train_loader,fp16=True), PrefetchLoader(val_loader,fp16=True)


