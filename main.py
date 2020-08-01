import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

from functions.functions import *
from model.model import *

# HYPERPARAMETERS #
lr = 2e-3
epochs = 1001
r_seed = np.random.seed(24)             # Defining a default random seed.
n_steps = np.random.randint(64, 96)     # Number of step of each epoch

# TENSOR FORMAT #
b = 8
h = 32
w = 32
n_channel = 16
n_filter = Sobels().__len__()           # Number of sobels configurated.

# FOLDERS #
weights_folder = "./weights"
results_folder = "./training_output"
os.mkdir(results_folder) if not os.path.exists(results_folder) else ""

# MODEL PARAMETERS # 
L2 = nn.MSELoss()                       # MSE as loss function.
device = check_cuda()                   # Select CPU or CUDA if is available.
save_steps_time   = epochs // 3         # Save plots of growining steps 3 times during the training proccess.
save_outpus_time  = epochs // 10        # Save plots of final epoch 10 times during the training process.    

# DEFINING MODEL 
model = NCA(n_channel, n_filter).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=500, gamma=0.1)

# DEFINING Y (TARGET IMAGE)
target_img = "./input/owl.png"
target = batch_target(target_img, b).to(device)

# TRAINING PROCESS
for epoch in range(epochs):

  # DEFINING X OR GRID CELL    
  X = call_grid((b, n_channel, h, w)).to(device)

  # STEP PROCESS
  for step in range(n_steps):
    X = model.forward(X)
    X = torch.clamp(X, 0, 1)

    # EXPORTING: SEQUENCE OF GROWNING CELL
    if epoch % save_steps_time == 0:
          step_folder = "steps_epoch_{}".format(epoch)
          steps_folder = os.path.join(results_folder, step_folder)
          os.mkdir(steps_folder) if not os.path.exists(steps_folder) else ""

          step_file = '{}_{}.png'.format(step, n_steps)
          step_path = os.path.join(steps_folder, step_file)

          fig = plt.figure()
          plt.imshow(unbatch_tensor(X))
          plt.savefig(step_path, bbox_inches='tight', dpi=100)
          plt.close(fig)

  # RUNNING PREDICTION      
  optimizer.zero_grad()
  X = torch.clamp(X, 0, 1)
  output = X[:, :4, :, :]

  # CALCULATING LOSS
  loss = L2(output, target)
  print("Epoch {}/{} - Loss: {}".format(epoch, epochs, str(loss.item())[0:6]))

  # EXPORTING: PREDICTION IMAGE
  if epoch % save_outpus_time == 0:

        fig = plt.figure()
        output_file = 'epoch_{}_loss_{}.png'.format(epoch, str(loss.item())[0:6])
        output_path = os.path.join(results_folder, output_file)
        plt.imshow(unbatch_tensor(output))
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close(fig)

  # OPTIMIZING PROCESS
  loss.backward()
  optimizer.step()
  scheduler.step()

# SAVING TRAINING
uid = gen_id()
weight_file = "nca-{}_epoch{}_loss{}.path".format(uid, epochs, str(loss.item())[0:6])

weights_path = os.path.join(weights_folder, weight_file)
torch.save(model.state_dict(), weights_path)
print("Model Saved.")