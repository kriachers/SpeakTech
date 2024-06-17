# =============================
# Section 1: Library imports
# =============================

import torch
import torchaudio
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import requests
import shutil
import yaml
from speechbrain.inference.classifiers import EncoderClassifier
import torch.nn as nn
import torch.optim as optim


# =============================
# Section 2: Data paths
# =============================

current_directory = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(current_directory, 'italian_big', 'corpus', 'it', 'train.tsv')
TEST_PATH = os.path.join(current_directory, 'italian_big', 'corpus', 'it', 'test.tsv')
DEV_PATH = os.path.join(current_directory, 'italian_big', 'corpus', 'it', 'dev.tsv')
AUDIO_FOLDER_PATH = os.path.join(current_directory, 'italian_big', 'corpus', 'it', 'clips')

train_df = pd.read_csv(TRAIN_PATH, sep='\t')
test_df = pd.read_csv(TEST_PATH, sep='\t')
dev_df = pd.read_csv(DEV_PATH, sep='\t')

# =============================
# Section 3: Download Model Files
# =============================

model_dir = os.path.join(current_directory, "pretrained_models", "ecapa")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# URLs of the files to download
files_to_download = {
    "hyperparams.yaml": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/hyperparams.yaml",
    "embedding_model.ckpt": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt"
}

# Download the files
for filename, url in files_to_download.items():
    response = requests.get(url, stream=True)
    file_path = os.path.join(model_dir, filename)
    with open(file_path, "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

# =============================
# Section 4: Load the Model Using Symlinks
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=model_dir, run_opts={"device": device})

# =============================
# Section 5: Data preparation
# =============================

# Function to extract ECAPA-TDNN features from an audio file
def extract_features(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embeddings = classifier.encode_batch(signal.to(device))
    return embeddings.squeeze().detach().cpu().numpy()

# Function to load data from dataframe with a limit on the number of items
def load_data(df, clips_folder, max_items=1000):
    data = []
    labels = []
    count = 0
    for idx, row in df.iterrows():
        if count >= max_items:
            break
        audio_path = os.path.join(clips_folder, f"{row['path']}")
        if os.path.exists(audio_path):
            features = extract_features(audio_path)
            data.append(features)
            labels.append(row['accents'])
            print(f"File {audio_path} found")
            count += 1
    return np.array(data), np.array(labels)

# =============================
# Section 6: Extract features and labels
# =============================

X_train, y_train = load_data(train_df, AUDIO_FOLDER_PATH, max_items=1000)
X_test, y_test = load_data(test_df, AUDIO_FOLDER_PATH, max_items=200)
X_dev, y_dev = load_data(dev_df, AUDIO_FOLDER_PATH, max_items=200)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_dev = label_encoder.transform(y_dev)

_ACCENTS_IT = label_encoder.classes_

# =============================
# Section 7: Create data loaders
# =============================

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
dev_dataset = TensorDataset(torch.tensor(X_dev, dtype=torch.float32), torch.tensor(y_dev, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

# =============================
# Section 8: Model setup
# =============================

class AccentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AccentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_dim = X_train.shape[1]
num_classes = len(_ACCENTS_IT)
model = AccentClassifier(input_dim, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================
# Section 9: Training loop
# =============================

def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

train(model, train_loader, criterion, optimizer, epochs=10)

# =============================
# Section 10: Evaluation
# =============================

def evaluate(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=_ACCENTS_IT))
    print(confusion_matrix(y_true, y_pred))

# =============================
# Section 11: Evaluation on test set
# =============================

evaluate(model, test_loader)