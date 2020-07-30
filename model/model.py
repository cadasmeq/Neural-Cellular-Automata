import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from functions.functions import check_cuda,perception_2, stochastic_update, detect_alives_cells

device = check_cuda()

class NCA(nn.Module):
    def __init__(self, n_channel, n_filter):
        super(NCA, self).__init__()
        self.n_channel = n_channel
        self.n_filter = n_filter
        self.fc1 = nn.Conv2d((self.n_channel * self.n_filter), 128, (1,1))
        self.fc2 = nn.Conv2d(128, n_channel, (1,1))
        torch.nn.init.zeros_(self.fc2.weight)
        #torch.nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
      b, c, h, w = x.size()
      
      #pVector = perception(x)  
      pVector = perception_2(x)  
      dx = self.fc1(pVector)
      dx = F.relu(dx) 
      dx = self.fc2(dx)      
      
      random_matrix = torch.from_numpy(np.random.randint(0, 2, (b, c, h, w))).to(device)
      #random_matrix = stochastic_update(x, dx)
      x = x + random_matrix * dx
      alives = detect_alives_cells(x)        # Conv (3,3) > 0.1

      return alives * x