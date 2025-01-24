import torch
import timm
timm.layers.set_fused_attn(False) ##n eeded to extract VORTEX features properly
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from RNN import RAE
import multiprocessing
import numpy as np
import random
import sys

class VORTEX(torch.nn.Module):    
    def __init__(self, backbone_name, input_size, pos_encoding=False, Q=1, norm=True, M=16): 
        super(VORTEX, self).__init__() 
        self.norm = norm
        self.M = M
              
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0) 
        self.backbone.eval()    
                  
        _, self.activation_shape = self.backbone.forward_intermediates(torch.zeros(2,3,input_size,input_size))
        self.depth = len(self.activation_shape)
               
        self.reshape = Rearrange('b l d w h -> b (l w h) d') 
        self.pre_process = lp_norm_layer(p=2.0, dim=1, eps=1e-10)
        
        self.activation_shape  = self.reshape(torch.stack(self.activation_shape , dim=1)).shape           
                           
        self.feature_encoders = [RAE(Q=Q, P=self.activation_shape[2], N=self.activation_shape[1], seed=i*Q*self.activation_shape[2],
                                    pos_encoding=pos_encoding, ntks = self.activation_shape[1]/self.depth, depth=self.depth)
                                    for i in range(int(self.M))]   
        
         
    def to(self, device):
        super(VORTEX, self).to(device)
        for encoder in range(len(self.feature_encoders)):
            self.feature_encoders[encoder].to(device)                
    
    def forward(self, x):               
        _, x  = self.backbone.forward_intermediates(x,indices=None) 
        x = torch.stack(x, dim=1)         
        x = self.reshape(x)
    
        if self.norm:
            x = self.pre_process(x)  

        return torch.sum(torch.stack([self.feature_encoders[i](x) for i in range(len(self.feature_encoders))], dim=2), dim=2)
    
    
class lp_norm_layer(torch.nn.Module):
    def __init__(self,p=2.0, dim=(1,2), eps=1e-10):
        super().__init__()   
        self.p=p
        self.dim=dim
        self.eps=eps

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=self.p, dim=self.dim, eps=self.eps)
  
  
class vanilla_featureAGG(torch.nn.Module):    
    def __init__(self, backbone_name, depth='last'): 
        super(vanilla_featureAGG, self).__init__()        
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0) 
        self.backbone.eval()
        self.depth = depth                                         

    def forward(self, x): 
        _, x = self.backbone.forward_intermediates(x)    
        if self.depth =='all':
            x = torch.stack(x, dim=1)  
            # print(x.shape)    
            x = torch.mean(torch.mean(x, dim=(3,4)), dim=1) 

        else: #if depth =='last'
            x = x[-1]
            x = torch.mean(x, dim=(2,3))

        return x  
    
       
def extract_features(backbone, dataset, pooling, seed, multigpu=False, batch_size=1, M=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
        
    total_cores=multiprocessing.cpu_count()
    num_workers = total_cores

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Creating the model
    if 'VORTEX' in pooling: #aggregative RAE 
        model = VORTEX(backbone, M=M, norm=True, pos_encoding=False, Q=1, features='activations')    
               
    elif pooling == 'GAP': 
        model = vanilla_featureAGG(backbone) 
    elif pooling == 'token':         
        model = timm.create_model(backbone, pretrained=True, num_classes=0)   
    else:
        raise ValueError(pooling + ' is an invalid ViT feature pooling method. Use VORTEX, GAP, or token.')
      
    model.to(device)
    
    data_loader = torch.utils.data.DataLoader(dataset,
                          batch_size=batch_size, shuffle = False, drop_last=False, pin_memory=True, num_workers=num_workers)  
   
    feature_size = model(dataset[0][0].unsqueeze(0).to(device)).cpu().detach().numpy().shape[1]
    if feature_size > 20000:
        raise ValueError('backbone returned too many features') #ignoring backbones with huge number of features
    
    print('extracting', feature_size, 'features for', len(dataset), 'images...')
    X = np.empty((0,feature_size))
    Y = np.empty((0))
        
    if multigpu:
        model.net = torch.nn.DataParallel(model.net)
        
    for i, data in enumerate(data_loader, 0):
      
        inputs, labels = data[0].to(device), data[1]   
              
        X = np.vstack((X,model(inputs).cpu().detach().numpy()))

        Y = np.hstack((Y, labels))

    del model
    del data_loader
    del dataset
    return X,Y