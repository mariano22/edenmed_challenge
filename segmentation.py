import argparse
import os

import torch
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.models import VGG11_Weights

from tools import get_torch_device

class Block(torch.nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = torch.nn.functional.relu(x, inplace=True)
        return out

def blend(origin, mask1=None, mask2=None):
    img = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
    if mask1 is not None:
        mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros_like(origin),
            torch.stack([mask1.float()]),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask1, 0.2)
        
    if mask2 is not None:
        mask2 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.stack([mask2.float()]),
            torch.zeros_like(origin),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask2, 0.2)
    
    return img

class PretrainedUNet(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
    
    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)
    
    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode
        
        self.init_conv = torch.nn.Conv2d(in_channels, 3, 1)
        
        endcoder = torchvision.models.vgg11(weights=VGG11_Weights.DEFAULT).features
        self.conv1 = endcoder[0]   # 64
        self.conv2 = endcoder[3]   # 128
        self.conv3 = endcoder[6]   # 256
        self.conv3s = endcoder[8]  # 256
        self.conv4 = endcoder[11]   # 512
        self.conv4s = endcoder[13]  # 512
        self.conv5 = endcoder[16]  # 512
        self.conv5s = endcoder[18] # 512
    
        self.center = Block(512, 512, 256, batch_norm)
        
        self.dec5 = Block(512 + 256, 512, 256, batch_norm)
        self.dec4 = Block(512 + 256, 512, 128, batch_norm)
        self.dec3 = Block(256 + 128, 256, 64, batch_norm)
        self.dec2 = Block(128 + 64, 128, 32, batch_norm)
        self.dec1 = Block(64 + 32, 64, 32, batch_norm)
        
        self.out = torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def forward(self, x):  
        init_conv = torch.nn.functional.relu(self.init_conv(x), inplace=True)

        enc1 = torch.nn.functional.relu(self.conv1(init_conv), inplace=True)
        enc2 = torch.nn.functional.relu(self.conv2(self.down(enc1)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3(self.down(enc2)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3s(enc3), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4(self.down(enc3)), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4s(enc4), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5(self.down(enc4)), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5s(enc5), inplace=True)
        
        center = self.center(self.down(enc5))
        
        dec5 = self.dec5(torch.cat([self.up(center, enc5.size()[-2:]), enc5], 1))
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))
        
        out = self.out(dec1)
        
        return out
    
def get_base_model():
    return PretrainedUNet(
            in_channels=1,
            out_channels=2, 
            batch_norm=True, 
            upscale_mode="bilinear"
        )

def load_model(model_path):
    device = get_torch_device()
    unet = get_base_model()
    unet.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    unet.to(device)
    unet.eval()
    return unet

def predict(filename, model):
    device = get_torch_device()
    origin_img = Image.open(filename).convert("P")
    original_size = origin_img.size  # (width, height)
    
    # Step 2: Resize and preprocess
    origin_resized = TF.resize(origin_img, (512, 512))
    origin_tensor = TF.to_tensor(origin_resized) - 0.5
    
    # Step 3: Run through model
    with torch.no_grad():
        input_tensor = origin_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(input_tensor)
    
        # Step 4: Get predicted mask
        softmax = F.log_softmax(output, dim=1)
        pred = torch.argmax(softmax, dim=1)  # shape: (1, H, W)
    
        # Step 5: Resize predicted mask to original size
        pred = pred.unsqueeze(1).float()  # Add channel dimension, shape: (1, 1, H, W)
        pred_resized = F.interpolate(pred, size=(original_size[1], original_size[0]), mode='nearest')  # size: (1, 1, H_orig, W_orig)
        pred_resized = pred_resized.squeeze().to("cpu").byte()  # shape: (H_orig, W_orig)

    mask_img = to_pil_image(pred_resized*255)
    return mask_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply the Lung Segmentation Model")
    parser.add_argument('--image', type=str, required=True, help='Path to the image')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to the trained model (.pt)')
    parser.add_argument('--out', type=str, required=True, help='Path to the output mask')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        exit(0)
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        exit(0)

    model = load_model(args.model)
    msk = predict(args.image, model)
    msk.save(args.out)
    print("Segmentation finished!")
    print(f"Mask was generated: {args.out}")
    

        