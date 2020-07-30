import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR

import os
import cv2
import uuid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_model():
    return ""

def load_model(model, weight_folder):
    model.load_state_dict(torch.load(weight_folder))
    model.eval()
    return "Model loaded."

# CUDA
def check_cuda():
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device("cuda:0")

    return device

device = check_cuda()

# GENERATE RANDOM ID
def gen_id():
    return str(uuid.uuid4().fields[-1])[:6]

# GENERATE X
def call_grid(size):
    x = torch.zeros(size).type(torch.float32)
    b, c, h, w = size
    x[:, 3:, h // 2, w // 2] = 1.0
    return x

# SOBEL CLASS
class Sobels():
    def sobels(self):
        identify = torch.tensor([[ 0,0,0],[0,1,0], [0,0,0]])
        dx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]])
        dy = torch.tensor([[ 1,2,1],[0,0,0], [-1,-2,-1]])
        return identify, dx, dy

    def batch_sobels(self, n_channel):
        self.n_channel = n_channel
        i, x, y = self.sobels()
        identify = i.repeat(self.n_channel, 1, 1, 1).type(torch.float32).to(device)
        dx = x.repeat(self.n_channel, 1, 1, 1).type(torch.float32).to(device)
        dy = y.repeat(self.n_channel, 1, 1, 1).type(torch.float32).to(device)

        return identify, dx, dy
    
    def __len__(self):
        return len(self.sobels())

# CREATE PERCEPTION
def perception(x):
    b,c,h,w = x.size()
    sobels = Sobels(c)
    sobel_I, sobel_X, sobel_Y = Sobels.get(c)
    c_sobel_I = F.conv2d(x, sobel_I, padding=1, groups=c)
    c_sobel_X = F.conv2d(x, sobel_X, padding=1, groups=c)
    c_sobel_Y = F.conv2d(x, sobel_Y, padding=1, groups=c)
    
    pv = torch.stack((c_sobel_I, c_sobel_X, c_sobel_Y)).type(torch.float32).view(b, n_filter*c, h, w) # view en lugar de shape ?
    return pv

def perception_2(x):
    b,c,h,w = x.size()
    sobels = Sobels()
    sobel_I, sobel_X, sobel_Y = sobels.batch_sobels(c)
    filters = [sobel_I, sobel_X, sobel_Y]
    perception = torch.empty((b, len(filters) * c, h, w)).to(device)

    # Computamos los vectores de percepción con cada filtro. 3 filtros x 16 = 48 componentes.
    for f, filt in enumerate(filters):
        perception[:, (f * c):((f+1) * c), :, :] = F.conv2d(x, filt, groups=c, padding=[1, 1])
    return perception

# GENERATE A STOCASTIC UPDATE
def stochastic_update(x, output):
    b,c,h,w = x.size()
    stochastic_matrix = torch.randint(0,2,((b,c,h,w))).to(device)  # 0 a 2 ?
    return stochastic_matrix

# DETECT ALIVE CELLS 
def detect_alives_cells(output):
    b,c,h,w = output.size()
    alive_filter = torch.ones((1,1,3,3)).type(torch.double).to(device)    # b = 1 ?
    alpha = (output[:,3:4,:,:] > 0.1).type(torch.double).to(device)
    alives = F.conv2d(alpha, alive_filter, padding=1)
    alives = (alives > 0.0)
    alives = alives.repeat(1, c, 1, 1)
    return alives

# BATCH TARGET IMAGE
def batch_target(file_path, batch):
    #img = cv2.imread(file_path, -1) / 255.0
    img = np.array(Image.open(file_path)) / 255.
    target = torch.from_numpy(img)[None, :, :, :].permute(0,3,1,2).type(torch.float32).to(device)
    target = target.repeat(batch,1,1,1)
    return target

# UNBATCH TENSOR   
def unbatch_tensor(x, default="Tensor"):
    b, c, h, w = x.size()
    x = x.permute(0,2,3,1)
    x = x.detach().cpu().numpy()
    return x[0,:,:,:4]

# PLOT A TENSOR
def plot_tensor(x, title, default="Tensor"):
    if default == "Tensor":
        b, c, h, w = x.size()
        x = x.permute(0,2,3,1)
        x = x.detach().cpu().numpy()
        plt.title(str(title).upper())
        plt.imshow(x[0,:,:,:4])

    elif default == "Image":
        img = cv2.imread(x, -1)
        plt.title(str(title).upper())
        plt.imshow(img)

# DEBUG TENSOR MIN MAX 
def tensorMinMax(x, name, debug):
    if debug:
        print("[{0}]\tMin: {1}, Max: {2}".format(name.capitalize(), x.min(), x.max()))
    else:
        pass