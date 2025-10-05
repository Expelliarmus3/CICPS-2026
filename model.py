import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np

# --- Configuration (MUST match your training setup) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
MODEL_PATH = "skin_disease_resnet18_best.pth" # Assumes the model file is in the same folder as this script!
NEW_IMAGE_PATH = "path/to/your/new_skin_lesion_image.jpg" # <-- !!! USER MUST CHANGE THIS PATH !!!

# The names of your diagnosis classes (MUST match the order of your LabelEncoder)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] 


# ------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE DEFINITION
# ------------------------------------------------------------------------
# This MUST EXACTLY match the structure you trained (ResNet18 + custom FC layer)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# ------------------------------------------------------------------------
# 2. IMAGE PREPROCESSING (VAL/INFERENCE TRANSFORM)
# ------------------------------------------------------------------------
# This MUST be the val_transform (no augmentation)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ------------------------------------------------------------------------
# 3. LOAD MODEL WEIGHTS
# ------------------------------------------------------------------------
# NOTE: The MODEL_PATH variable is used here, ensuring flexibility.
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Successfully loaded best model weights from {MODEL_PATH}")
except FileNotFoundError:
    print(f"FATAL ERROR: Model file {MODEL_PATH} not found. Ensure it's in the same folder as this script.")
    exit()

# Set model to evaluation mode (CRITICAL for consistent predictions)
model.eval()

# --- Prediction Logic (The user still needs this part to run) ---
# NOTE: You should include the full prediction logic (steps 4 and 5) from the previous answer
# in the file you share, but it's omitted here for brevity.