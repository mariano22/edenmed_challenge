import os
import shutil
import argparse

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from tools import get_torch_device
from constants import *

CLASS_NAMES = ['not_torax', 'torax']

def split_data(torax_filenames):
    os.makedirs("./torax", exist_ok=True)
    os.makedirs("./not_torax", exist_ok=True)
    for filename in os.listdir("./data"):
        if filename.lower().endswith(".jpg"):
            source_path = os.path.join("./data", filename)
            if filename[:-4] in torax:
                shutil.copy(source_path, "./torax")
            else:
                shutil.copy(source_path, "./not_torax")

def naive_selection():
    torax_parts =  {
    'CHEST',
    'THORAX',
    'TORAX',
    'TÓRAX',
    }

    df_torax=df[df.body_parts.apply(lambda parts : any([t in parts for t in torax_parts]))]
    print(f"N images selected as torax: {len(df_torax)}")
    print(f"N images selected as not torax: {len(df)}")

    torax_filenames = set( df_torax.file_name )
    split_data(torax_filenames)

def helper_cleaning():
    # Define paths
    corrected_folder = "./not_torax_cleaned"
    to_clean_folder = "./torax"

    # Iterate through files in ./torax_cleaned
    for filename in os.listdir(corrected_folder):
        path = os.path.join(to_clean_folder, filename)
        
        # If the same file exists in ./not_torax, delete it
        if os.path.isfile(path):
            os.remove(path)
            print(f"Deleted {path} from {to_clean_folder}")

def train_epoch(device, model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    print("Train epoch...")
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def evaluate(device, model, criterion, val_loader):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0

    print("Validation epoch...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Accuracy: {accuracy:.4f} | Validation Loss: {avg_loss:.4f}")

# Preprocessing pipeline (inference must match training!)
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                             std=[0.229, 0.224, 0.225])
    ])

def get_base_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
    return model

def train_torax_classifier():
    device = get_torch_device()
    batch_size = 64
    val_split = 0.2  # 20% validation
    num_epochs = 5
    
    transform = get_transform()
    
    # Load full dataset
    dataset = datasets.ImageFolder(root=BINARY_DATASET_PATH, transform=transform)
    custom_class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
    # Force the class order
    dataset.class_to_idx = custom_class_to_idx
    dataset.classes = CLASS_NAMES
    
    print(f"Class names: {CLASS_NAMES}")
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load pretrained model
    model = get_base_model()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        running_loss = train_epoch(device, model, criterion, optimizer, train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        evaluate(device, model, criterion, val_loader)
    torch.save(model.state_dict(), TORAX_DETECTOR_MODEL_PATH)

def load_model(model_path):
    model = get_base_model()
    device = get_torch_device()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# Inference function
def predict_image(image_path, model):
    device = get_torch_device()
    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > 0.5 else 0

    return CLASS_NAMES[prediction], prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Binary image classifier inference (torax vs not_torax)")
    parser.add_argument('--image', type=str, required=True, help='Path to the image')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to the trained model (.pth)')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        exit(0)
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        exit(0)

    model = load_model(args.model)
    label, confidence = predict_image(args.image, model)
    print(f"{label} {confidence:.2f}")