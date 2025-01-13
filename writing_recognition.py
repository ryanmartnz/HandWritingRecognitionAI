"""
File: project5.py
Author: Ryan Martinez
Date: 12/06/2024
Description: This program analyzes an image of handwritten words and outputs the words detected on the image. Each letter in the word 
    is expected to be at most 45x45 pixels large.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

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

# Dataset class definition
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx,0]/255.0)[np.newaxis].astype(np.float32)

# Classes list to hold each possible detected letter
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Define model and load model from file
model = LeNet(len(classes))
model.load_state_dict(torch.load('./model.pth', weights_only=False))

# Get the input image from the input
input_image = input("Enter the name of the image to read: ")

# Read the image
img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

# OTSU thresholding on the image
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Define structuring element, (3,3) is small enough the detect the letters
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Apply dilation on the threshold image
dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)

# Finding contours
contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# test_images array holds the images of each letter on the image
test_images = []

# Arrays to hold the bounding boxes of each letter 
bounding_boxes = []

# Loop through the contours
for contour in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Append the bounding rectangle to the array
    bounding_boxes.append((x, y, w, h))

    # Crop the image around the letter
    cropped = img[y:y + h, x:x + w]
    
    # The Model accepts 45x45 images of each letter
    # Since the bounding boxes of the letters will be smaller than that,
    # I must add padding to the images
    og_height, og_width = cropped.shape[:2]

    # Calculate the needed padding on the image
    needed_height = 45 - og_height
    needed_width = 45 - og_width

    top = needed_height // 2
    bottom = needed_height // 2

    if needed_height % 2 != 0:
        bottom += 1

    left = needed_width // 2 
    right = needed_width // 2
    if needed_width % 2 != 0:
        right += 1

    # Add the padding to the image
    image = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])

    # Resize and flatten the image 
    image = cv2.resize(image, (45, 45))
    image_array = np.array(image) / 255.0 

    test_images.append(image_array)

# Reshape the images in the array to be the expected size
test_images = np.array(test_images)
bs, h, w = test_images.shape
test_images = (test_images.reshape(bs, 1, h, w)*255.0).astype(np.uint8)

# Create the dataset and dataloader for the images
dataset = TestDataset(test_images)
testloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# Array to hold the predicitions for each letter
predictions = []

# Get the model's predictions on the letters in the image
model.eval()
with torch.no_grad():
    for batch in testloader:
        output = model(batch)
        pred = torch.argmax(output, dim=1)
        predictions = [classes[p] for p in pred.tolist()]

# Appending the predicition of each letter to its bounding box
boundbox_pred = []
for i in range(len(bounding_boxes)):
    x,y,w,h = bounding_boxes[i]
    pred = predictions[i]
    boundbox_pred.append((x,y,w,h,pred))

# Sort the letters based on their y-coordinate
sorted_indices = sorted(range(len(boundbox_pred)), key=lambda i: boundbox_pred[i][1])
boundbox_pred_sorted = [boundbox_pred[i] for i in sorted_indices]

# Begin sorting the letters based on their y-coordinate
# This is to seperate the lines of words in the image
groups = []
current_group = [boundbox_pred_sorted[0]]

# Loop through the letters
for (x,y,w,h,pred) in boundbox_pred_sorted[1:]:
    cur_x,cur_y,cur_w,cur_h,cur_pred = current_group[-1]

    # If two letters are close enough vertically, add the current letter to the current group
    if abs((y + (h//2)) - (cur_y + (cur_h//2))) <= 17:
        # new_y = cur_y
        current_group.append((x,y,w,h,pred))
    # If two letters are far enough away vertically, sort the current group based on the x-coordinate, add it to the list, and create a new group
    else:
        sorted_group = sorted(current_group, key=lambda coord: coord[0])
        groups.append(sorted_group)
        current_group = [(x,y,w,h,pred)]

# Sort the last group and append the group to the list
sorted_group = sorted(current_group, key=lambda coord: coord[0])
groups.append(sorted_group)

# Now the letters are separated into a 2-d array where each array is a line on the image,
# and each element of the arrays are the letters in each line in order from left to right
# Now I must seperate these letters based on the word

# These arrays hold the sentences and the current word in the loop
sentences = []
words = []

# Loop through the groups
for i in range(len(groups)):
    # keep track of the current word 
    current_word = [groups[i][0]]
    # Loop through the letters in the current group
    for (x,y,w,h,pred) in groups[i][1:]:
        cur_x,cur_y,cur_w,cur_h,cur_pred = current_word[-1]

        # If two letters in the group are close enough, append the letter to the current word
        if abs((cur_x+cur_w) - x) < 14:
            current_word.append((x,y,w,h,pred))
        # If two letters in the group are far away enough, add the current word to the words list and start a new word
        else:
            words.append(current_word)
            current_word = [(x,y,w,h,pred)]
    
    # If there are no more letters in the current group, add the current word to the words list and append the words to the sentences list
    words.append(current_word)
    sentences.append(words)
    # restart grouping words
    words = []
    current_word = [groups[i][0]]

# Output the words detected on the image
print("\nWords detected on the image: ")
for sentence in sentences:
    for word in sentence:
        for letter in word:
            print(letter[-1], end="")
        print(" ", end="")
    print()
