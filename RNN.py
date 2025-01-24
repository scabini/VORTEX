import math
import torch
import pickle


class RAE(torch.nn.Module):
    def __init__(self, Q, P, N, pos_encoding, seed,  ntks=14*14, depth=12):
        super(RAE, self).__init__() 
        self.pos_encoding=pos_encoding
        if self.pos_encoding:
            window_size = math.sqrt(ntks)
            enc2d = positionalencoding2d(int(P), int(window_size), int(window_size)).reshape(int(P), int(ntks)).T
            self._encoding = torch.nn.Parameter(torch.vstack([enc2d #* ((i+1)/depth)
                                                for i in range(depth)]))
        
        self.hidden = torch.nn.Linear(in_features=P, out_features=Q, bias=True)
        self.hidden.weight = torch.nn.Parameter(make_orthogonal(LCG(Q, P, seed)), requires_grad=False)
        self.hidden.bias = torch.nn.Parameter(torch.ones(Q), requires_grad=False)
        self._activation = torch.sigmoid
        
            
    def forward(self, x):     
        if self.pos_encoding:
            for batch in range(len(x)):
                x[batch] = torch.add(x[batch], self._encoding)
        
        H = self._activation(self.hidden(x))
        
        x = torch.linalg.lstsq(H, x).solution

        return x.reshape(x.shape[0], x.shape[1]*x.shape[2])
   
    
        
def LCG(m, n, seed):
    L = m*n
    if L == 1:
        return torch.ones((1,1), dtype=torch.float)
    else:      
        with open('RAE_LCG_weights.pkl', 'rb') as f:
            V = pickle.load(f)
            f.close()      
        V = V[seed:L+seed]
              
        ##### If you want to generate the weights everytime, instead of loading
        #       from our file, just uncomment these lines below and remove the above ones
        
        # V = torch.zeros(L, dtype=torch.float)    
        # V[0] = 0
        # a = 75
        # b = 74   
        # c = (2**16)+1        
        # for x in range(1, (m*n)):
        #   V[x] = (a*V[x-1]+b) % c

          
        #Always keep the zscore normalization for our LCG weights
        V = torch.divide(torch.subtract(V, torch.mean(V)), torch.std(V))
 
    return V.reshape((m,n))
    
def make_orthogonal(tensor):
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = torch.reshape(tensor, (rows, cols))

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()
        
    return q

##Not used on the VORTEX paper, but can be added for taks where patch position is important (activate it by setting pos_encoding=True)
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    # if d_model % 4 != 0:
    #     raise ValueError("Cannot use sin/cos positional encoding with "
    #                      "odd dimension (got dim={:d})".format(d_model))
    d_model_orig = d_model
    if d_model % 4 != 0:
        d_model = d_model+2   # Round up to the nearest multiple of 4
    else:
        d_model = d_model_orig    
        
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe[:d_model_orig, :, :]
