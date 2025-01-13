"""
File: model.py
Author: Ryan Martinez
Date: 12/06/2024
Description: This program defines and trains a CNN model to classify handwritten letters of the English alphabet into lowercase 
    and uppercase letters. The images are expected to be 45x45 pixels.
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

# Transformations to make randomly on test images to improve predictions
aug_transforms = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.RandomResizedCrop((45,45), (0.8,1.0), (0.9,1.1)),
    torchvision.transforms.ColorJitter(0.1,0.1,0.1,0.1)
)

# Dataset class definition
class LetterDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode

        images = []
        labels = []

        if mode == 'train':
            # Load training images and labels
            for i in range(1, 243):
                og = cv2.imread(f"./train_model/letters/{i}.png", cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(og, (45, 45))
                image_array = np.array(image) / 255.0 
                images.append(image_array)

                with open(f"./train_model/letters/{i}.txt", "r") as f:
                    char = f.readline()
                    char = char[0]
                    class_num = classes.index(char)
                
                labels.append(class_num)
            
            for i in range(1, 313):
                og = cv2.imread(f"./train_model/alphabet/{i}.png", cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(og, (45, 45))
                image_array = np.array(image) / 255.0 
                images.append(image_array)

                with open(f"./train_model/alphabet/{i}.txt", "r") as f:
                    char = f.readline()
                    char = char[0]
                    class_num = classes.index(char)
                
                labels.append(class_num)
        
        elif mode == 'test':
            # Load testing images and label
            for i in range(1, 113):
                og = cv2.imread(f"./test_model/{i}.png", cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(og, (45, 45))
                image_array = np.array(image) / 255.0 
                images.append(image_array)

                with open(f"./test_model/{i}.txt", "r") as f:
                    char = f.readline()
                    char = char[0]
                    class_num = classes.index(char)
                
                labels.append(class_num)

        images = np.array(images)
        labels = np.array(labels)

        x = images
        self.y = labels
        
        # Reshape images to required size
        bs, h, w = x.shape
        self.x = (x.reshape(bs, 1, h, w)*255.0).astype(np.uint8)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # If training, randomly apply some transformation to images to improve prediction
            x = (np.asarray(aug_transforms(Image.fromarray(self.x[idx,0])))/255.0)[np.newaxis].astype(np.float32)
        else:
            x = (self.x[idx,0]/255.0)[np.newaxis].astype(np.float32)
        return x, self.y[idx]
    
# CNN model class definition
class LeNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 3 convolutional layers for progressive feature extraction
        # First layer to capture low-level features like edges and corners
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second layer will combine the low-level features into more complex patterns, such as curves or parts of a letter
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        # Third layer captures even more abstract features, like combinations of shapes
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d((2,2))
        # First linear layer compresses the extracted features into more compact, meaningful representations
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        # Second linear layer takes the compressed features and reduces them further
        self.fc2 = nn.Linear(256, 128)
        # Third linear layer maps the final abstract features to the number of output classes 
        self.fc3 = nn.Linear(128, num_classes)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 22x22
        x = self.pool(F.relu(self.conv2(x))) # 11x11
        x = self.pool(F.relu(self.conv3(x))) # 5x5
        x = torch.flatten(x, start_dim=1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Classes list to hold each possible detected letter
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

if __name__ == '__main__':
    # Create the training dataset and dataloader for the training data
    train_dataset = LetterDataset('train')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Create the testing dataset and dataloader for the testing data
    test_dataset = LetterDataset('test')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Define the CNN model and define the model optimizer and learning rate
    model = LeNet(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Keep track of best testing accuracy
    best_acc = 0.0
    # Patience of 1000 ensures highest accuracy possible is saved
    patience = 1000
    epochs = 0
    # Training Loop
    while epochs < patience:

        # Train the model
        model.train()
        for x, y in trainloader:
            optimizer.zero_grad()
            y_ = model(x)
            loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(y_, dim=1), y)
            loss.backward()
            optimizer.step()
        epochs += 1

        # Every ten epochs, evaluate the training and testing accuracy and loss
        if epochs%10 == 9:
            model.eval()
            with torch.no_grad():
                correct = 0
                for x, y in trainloader:
                    y_ = model(x)
                    y_ = torch.argmax(y_, 1)
                    correct += torch.sum(y_ == y)
                print('train loss: ', loss.item(), '/ train acc: ', correct/len(train_dataset))
            
            # evalue testing accuracy
            with torch.no_grad():
                correct = 0
                for x, y in testloader:
                    y_ = model(x)
                    y_ = torch.argmax(y_, 1)
                    correct += torch.sum(y_ == y)
                acc = correct/len(test_dataset)
                print('test acc: ', correct/len(test_dataset))

                # If the test accuracy is better than the current recorded best accuracy, update the best accuracy
                # Save the model, and reset the epochs
                if best_acc < acc:
                    best_acc = acc
                    torch.save(model.state_dict(), './model.pth')
                    epochs = 0

    # Output final training message
    print('Finished Training')
    print('Best accuracy: ', best_acc)