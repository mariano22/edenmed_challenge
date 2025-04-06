from pathlib import Path
import os
import pandas as pd
import json
from collections import OrderedDict
import time 
import argparse

import tarfile
import wget
import shutil
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from unet import UNet
from tools import *

SEGMENTATION_IMAGES_PATH = Path('data/segmentation_train_ds/images')
SEGMENTATION_MASK_PATH = Path('data/segmentation_train_ds/masks')
SEGMENTATION_SPLIT_FILE = "data/enmed_splits.pk"

DATASET_S3_URL = "https://enmed-lung-dataset.s3.us-east-1.amazonaws.com/dataset.tar.gz"

SHENZHEN_PREFIX = "CHNCXR"
MONTGOMERY_PREFIX = "MCUCXR"

BATCH_SIZE = 4
TRAIN_LOG_FILENAME = "logs/segmentation_train-log.txt"
MODELS_FOLDER = Path("models")
MODEL_NAME = 'unet-enmed.pt'

def download_dataset_if_not_exists():
    if os.path.exists(SEGMENTATION_MASK_PATH):
        print("Segmentation train dataset already exists locally")
    else:
        # Download
        segmentation_dataset_path = Path(SEGMENTATION_MASK_PATH).parent
        data_folder_path = segmentation_dataset_path.parent
        print(f"Segmentation train dataset doesn't exist {segmentation_dataset_path}")
        print("Downloading...")
        download_dst = data_folder_path / Path(DATASET_S3_URL).name
        print(download_dst)
        wget.download(DATASET_S3_URL, out=str(download_dst))
    
        # Untar
        with tarfile.open(download_dst, "r:gz") as tar:
            tar.extractall(path=segmentation_dataset_path)
    
        # Clenup
        for cls in ['images', 'masks']:
            shutil.move(segmentation_dataset_path / 'dataset' / cls,segmentation_dataset_path / cls)
        shutil.rmtree(segmentation_dataset_path / 'dataset' )
        os.remove(str(download_dst))

def load_filenames():
    download_dataset_if_not_exists()
    file_names = [f.stem for f in Path(SEGMENTATION_MASK_PATH).glob("*.png")]
    file_names = [x.replace("_mask", "") for x in file_names]
    return file_names

def load_filenames_splits():
    file_names = load_filenames()

    if os.path.isfile(SEGMENTATION_SPLIT_FILE):
        with open(SEGMENTATION_SPLIT_FILE, "rb") as f:
            splits = pickle.load(f)
    else:
        splits = dict()
        splits["train"] = [fname for fname in file_names if SHENZHEN_PREFIX in fname]
        splits["test"] = [fname for fname in file_names if MONTGOMERY_PREFIX in fname]
        splits["train"], splits["val"] = train_test_split(splits["train"],
                                                          test_size=0.15, random_state=42)

        with open(SEGMENTATION_SPLIT_FILE, "wb") as f:
            pickle.dump(splits, f)
    return splits

class LungDataset(torch.utils.data.Dataset):
    def __init__(self, file_names, image_transform=None, mask_transform=None):
        self.file_names = file_names
        self.image_transform = image_transform
        self.mask_transform = mask_transform
    
    def __getitem__(self, idx):
        fname = self.file_names[idx]
        
        image = torchvision.io.read_image(SEGMENTATION_IMAGES_PATH / (fname + ".png")).float()/255.0
        if image.shape[0]==1:  
            image = image.repeat(3,1,1)
        if self.image_transform is not None:
            image = self.image_transform(image)

        mask = torchvision.io.read_image(SEGMENTATION_MASK_PATH / (fname + ("_mask" if SHENZHEN_PREFIX in fname else "") + ".png")).float()/255.0
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        return image, mask 
    
    def __len__(self):
        return len(self.file_names)

def jaccard(y_true, y_pred):
    """ Jaccard a.k.a IoU score for batch of images
    """
    
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    intersection = (y_true_flat * y_pred_flat).sum(1)
    union = ((y_true_flat + y_pred_flat) > 0.0).float().sum(1)
    
    score = (intersection) / (union + eps)
    score = score.sum() / num
    return score
    

def dice(y_true, y_pred):
    """ Dice a.k.a f1 score for batch of images
    """
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    intersection = (y_true_flat * y_pred_flat).sum(1)
    
    score =  (2 * intersection) / (y_true_flat.sum(1) + y_pred_flat.sum(1) + eps)
    score = score.sum() / num
    return score

def train_epoch(device, datasets, dataloaders, unet, optimizer, criterion):
    unet.train()
    train_loss = 0
    for images, masks in tqdm(dataloaders["train"]):
        images = images.to(device)
        masks = masks.to(device).squeeze(1)
        
        optimizer.zero_grad()
        
        outs = unet(images)
        loss = criterion(outs, masks)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)    
    train_loss = train_loss / len(datasets['train'])
    return OrderedDict([
        ("train_loss", train_loss),
    ])

def valid_epoch(device, datasets, dataloaders, unet, criterion):
    unet.eval()
    val_loss = 0.0
    val_jaccard = 0.0
    val_dice = 0.0
    
    for images, masks in tqdm(dataloaders["val"]):
        batch_size = images.size(0)
    
        images = images.to(device)
        masks = masks.to(device).squeeze(1)
    
        with torch.no_grad():
            logits = unet(images)
            predictions = torch.argmax(logits, dim=1)
            
            val_loss += criterion(logits, masks).item() * batch_size
    
            masks = masks.float()
            predictions = predictions.float()
            
            val_jaccard += jaccard(masks, predictions).item() * batch_size
            val_dice += dice(masks, predictions).item() * batch_size
    
    
    val_loss = val_loss / len(datasets["val"])
    val_jaccard = val_jaccard / len(datasets["val"])
    val_dice = val_dice / len(datasets["val"])
    return OrderedDict([ 
            ("val_loss", val_loss),
            ("val_jaccard", val_jaccard),
            ("val_dice", val_dice), ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train our own Segmentation Model")
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train')
    args = parser.parse_args()

    splits = load_filenames_splits()
    
    image_tfms = transforms.Compose([ # We upscale the 1-channel images here so we use ImageNet mean/std normalize
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std because we are using VGG11_Weights
                                 std=[0.229, 0.224, 0.225])
        ])
    
    mask_tfms = transforms.Compose([
                transforms.Resize((512, 512)),
            ])

    datasets = {split_key: LungDataset(splits[split_key], image_tfms, mask_tfms) for split_key in splits}
    
    dataloaders = { split_key: DataLoader(datasets[split_key],
                                     batch_size=BATCH_SIZE,
                                     #num_workers=4, 
                                     shuffle=(split_key=="train"),
                                     drop_last=(split_key=="train")) for split_key in split }
    
    device = get_torch_device()
    
    unet = UNet(n_channels=3, n_classes=2, bilinear=True)
    unet = unet.to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_loss = np.inf
    history = []

    with open(TRAIN_LOG_FILENAME, "a") as train_log_file:
        def log(line):
            print(line)
            train_log_file.write(line + "\n")

        log(f"STARTING LOOP OF {EPOCHS} EPOCHS")
        for epoch in range(args.epochs):
            metrics = OrderedDict([("epoch", epoch)])
            
            start_t = time.time()
            metrics |= train_epoch(device, datasets, dataloaders, unet, optimizer, criterion)
            metrics |= valid_epoch(device, datasets, dataloaders, unet, criterion)
            end_t = time.time()
            metrics["time"] = end_t - start_t
            
            # Log metrics
            report = ' '.join([f"{name}={value:.4f}" for name,value in metrics.items()]) + '\n'
            log(report)
        
            # Keep the history of the metrics
            history.append(metrics)
        
            # Save the best version of the model
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                torch.save(unet.state_dict(), MODELS_FOLDER / MODEL_NAME)
                log("model saved")
    
    