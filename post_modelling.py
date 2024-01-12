import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from collections import Counter

class_names = ['Arm flapping', 'Finger flicking', 'Null']

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        # Load pre-trained VGG-16
        self.vgg = models.vgg16(pretrained=True).features
        # Freeze VGG features
        for param in self.vgg.parameters():
            param.requires_grad = False

        feature_map_size = 7 * 7 * 512

        # LSTM layer
        self.lstm = nn.LSTM(input_size=feature_map_size, hidden_size=256, num_layers=1, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(256, 3)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        c_out = self.vgg(c_in)

        # Flatten the output for LSTM input
        c_out = c_out.view(c_out.size(0), -1)

        # Reshape and pass through LSTM
        c_out = c_out.view(batch_size, timesteps, -1)
        lstm_out, _ = self.lstm(c_out)
        out = self.fc(lstm_out[:, -1, :])
        return out


# Function to load the model
def load_model(model_path):
    model = CNNLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to process and predict each bunch
def process_bunch_and_predict(model, bunch_folder, transform):
    bunch_frames = [os.path.join(bunch_folder, f) for f in sorted(os.listdir(bunch_folder)) if f.endswith('.jpg')]
    bunch_data = []

    for frame_path in bunch_frames:
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension
        bunch_data.append(image)

    bunch_data = torch.cat(bunch_data, dim=0).unsqueeze(0)
    predictions = model(bunch_data)
    predicted_classes = predictions.argmax(dim=1)  # Assuming your model outputs raw logits

    return predicted_classes

# Function to process the video and perform majority voting
def process_video_and_vote(model, video_folder, transform):
    bunch_predictions = []

    for bunch_folder in sorted(os.listdir(video_folder)):
        if os.path.isdir(os.path.join(video_folder, bunch_folder)):
            predictions = process_bunch_and_predict(model, os.path.join(video_folder, bunch_folder), transform)
            bunch_predictions.extend(predictions.tolist())

    print(bunch_predictions)

    # Apply majority voting logic
    if all(pred == 2 for pred in bunch_predictions):
        return 'Null'
    else:
        filtered_preds = [pred for pred in bunch_predictions if pred != 2]
        if filtered_preds:
            most_common_pred = Counter(filtered_preds).most_common(1)[0][0]
            return class_names[most_common_pred]
        else:
            return 'Null'


# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
