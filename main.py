import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from functions.functions import *
from model.model import *

plt.ion()

ebug_minMax = False
device = torch.device('cuda:0')

lr = 2e-3
epochs = 1001
n_filter = 3 

r_seed = np.random.seed(24)
n_steps = np.random.randint(64, 96)

b, n_channel, h, w = 8, 16, 32, 32
tensor_format = (b, n_channel, h, w)
plt_step = 100

# model
model = NCA(n_channel, n_filter).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=500, gamma=0.1)
L2 = nn.MSELoss()

img = ".input/owl.png"
target = batch_target(img, b).to(device)
x = call_grid(tensor_format).to(device)

for epoch in range(epochs):
      
  result = call_grid(tensor_format).to(device)

  for step in range(n_steps):
    result = model.forward(result)
    result = torch.clamp(result, 0, 1)
 
  optimizer.zero_grad()
  result = torch.clamp(result, 0, 1)
  output = result[:, :4, :, :]
  loss = L2(output, target)

  if epoch % plt_step == 0:
    plt.imshow(unbatch_tensor(output))
    plt.pause(1)

  loss = L2(output, target)
  print("Epoch {}/{} - Loss: {}".format(epoch, epochs, str(loss.item())[0:6]))
  loss.backward()
  optimizer.step()
  scheduler.step()