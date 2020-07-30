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

debug_minMax = False
device = torch.device('cuda:0')

lr = 2e-3
epochs = 100
n_filter = 3 

r_seed = np.random.seed(24)
n_steps = np.random.randint(64, 96)

b, n_channel, h, w = 8, 16, 32, 32
tensor_format = (b, n_channel, h, w)

save_steps_time   = epochs // 3
save_outpus_time  = epochs // 5

results_folder = "./training_output"
os.mkdir(results_folder) if not os.path.exists(results_folder) else ""

# model
model = NCA(n_channel, n_filter).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=500, gamma=0.1)
L2 = nn.MSELoss()

img = "./input/cynda.png"
target = batch_target(img, b).to(device)
x = call_grid(tensor_format).to(device)

for epoch in range(epochs):
      
  result = call_grid(tensor_format).to(device)

  for step in range(n_steps):
    result = model.forward(result)
    result = torch.clamp(result, 0, 1)

    if epoch % save_steps_time == 0:
          step_folder = "steps_epoch_{}".format(epoch)
          steps_folder = os.path.join(results_folder, step_folder)
          os.mkdir(steps_folder) if not os.path.exists(steps_folder) else ""

          step_file = 'epoch_{}_step{}_{}.png'.format(epoch, step, n_steps)
          step_path = os.path.join(steps_folder, step_file)

          fig = plt.figure()
          plt.imshow(unbatch_tensor(result))
          plt.savefig(step_path, bbox_inches='tight', dpi=100)
          plt.close(fig)
        
  optimizer.zero_grad()
  result = torch.clamp(result, 0, 1)
  output = result[:, :4, :, :]
  loss = L2(output, target)

  if epoch % save_outpus_time == 0:
        # save outpout
        fig = plt.figure()
        output_file = 'output_epoch_{}.png'.format(epoch)
        output_path = os.path.join(results_folder, output_file)
        plt.imshow(unbatch_tensor(output))
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close(fig)

  loss = L2(output, target)
  print("Epoch {}/{} - Loss: {}".format(epoch, epochs, str(loss.item())[0:6]))
  loss.backward()
  optimizer.step()
  scheduler.step()

# save model

uid = gen_id()
weights_folder = "./weights"
weight_file = "nca-{}_epoch{}_loss{}.path".format(uid, epochs, str(loss.item())[0:6])

weights_path = os.path.join(weights_folder, weight_file)
torch.save(model.state_dict(), weights_path)
print("Model Saved.")