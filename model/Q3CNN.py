import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import Cluster

# This is for the progress bar.
from tqdm.auto import tqdm
batch_size = 12

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
DATASET = "题目/B/附件"
Au_DATASET = DATASET + "/Au20_OPT_1000"
B_DATASET = DATASET + "/B45-_OPT_3751"


class ClusterDataset(Dataset):
    def __init__(self, element, mode='train', transform=None):
        self.transform = transform
        if element == 'Au':
            data_set = Au_DATASET
        elif element == 'B':
            data_set = B_DATASET
        else:
            raise ValueError
        self.data = []
        for file in os.listdir(data_set):
            c = Cluster(file)
            c.read_xyz(data_set)
            self.data.append(c)
        self.np_x = []
        self.np_y = []
        for cluster in self.data:
            self.np_x.append(cluster.seq(1))
            self.np_y.append(cluster.energy)
        self.np_x = np.array(self.np_x)
        self.np_y = np.array(self.np_y)
        if mode == 'train':
            self.np_x = self.np_x[:int(len(self.data) * 0.7)]
            self.np_y = self.np_y[:int(len(self.data) * 0.7)]
        elif mode == 'val':
            self.np_x = self.np_x[int(len(self.data) * 0.7):]
            self.np_y = self.np_y[int(len(self.data) * 0.7):]

    def __len__(self):
        return len(self.np_x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.np_x[idx]
        if self.transform:
            sample = self.transform(sample)

        return torch.DoubleTensor(sample), self.np_y[idx]


# Construct data loaders.
train_set = ClusterDataset('B')
val_set = ClusterDataset('B', 'val')

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.conv = nn.Conv2d(1, 64, 3, 1, 1)
        self.norm = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(2, 2, 0)
        self.fc_layers = nn.Sequential(
            nn.Linear(1408, 512), # 10 * 8 * 8
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pool(x)
        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
model = model.double()

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.MSELoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

# The number of training epochs.
n_epochs = 800

# Whether to do semi-supervised learning.
do_semi = False

for epoch in range(n_epochs):
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.unsqueeze(1)
        labels = labels.unsqueeze(1)
        bs = imgs.size(0)
        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(val_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.unsqueeze(1)
        labels = labels.unsqueeze(1)

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
