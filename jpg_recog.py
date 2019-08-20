import numpy as np
from cv2 import cv2
import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt

print(cv2.__version__)

class Net(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.lsm(x)
        return x

# Neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
model = Net(input_size, hidden_sizes, output_size)
print(model)

model.load_state_dict(torch.load('numpredic_98accuracy.pt'))
model.eval()

# Get the image 
frame = cv2.imread('numbers.jpg')
frame = cv2.resize(frame, (int(4032/5), int(3024/5)))

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholds the image
im_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Finds the contours 
contours, hierarchy = cv2.findContours(im_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Clean the contour noise
min_area = 1500     #threshold area
max_area = 2500
contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) and max_area > cv2.contourArea(cnt)]

# Draws the contours 
cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
    
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in contours]

# find the distance between the two points in rectangles
dist = [np.sqrt(rect[2]**2 + rect[3]**2) for rect in rects]

# Increase the size of rectangles by approximately 40%
# A 10% of the lenght is 
inc = [int(np.floor(d*0.2)) for d in dist]

ROIs = []
for i in range(len(rects)):
    p1 = rects[i][0] - inc[i], rects[i][1] - inc[i]
    p2 = rects[i][0] + rects[i][2] + inc[i], rects[i][1] + rects[i][3] + inc[i]
    cv2.rectangle(gray, p1, p2, (255, 255, 255), 3)
    roi = gray[p1[1]:p2[1], p1[0]:p2[0]]
    ROIs.append(roi)
# cv2.imshow('frame', gray)
# cv2.waitKey(1000)

# Resize the region of interest to 28x28
ROIs = [cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA) for roi in ROIs]

transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Pre-process image
raw_img = ROIs[0]
norm_img = cv2.normalize(raw_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
norm_img = 1 - norm_img
norm_img[norm_img<=0.2] = 0
tensor_img = transform(norm_img)

final_img = tensor_img

plt.imshow(final_img.numpy().squeeze(), cmap='gray_r')
plt.show()

final_img = final_img.view(final_img.shape[0], -1)

with torch.no_grad():
    output = model(final_img)
    probability = torch.exp(output)
    probability = probability[0].numpy()
    print(probability)
    prediction = np.argmax(probability)
print('The predicted number is {} with a probability of {:.4}%'.format(prediction, probability[prediction]*100))


cv2.waitKey(0) # wait for any key
cv2.destroyAllWindows() # close the image window