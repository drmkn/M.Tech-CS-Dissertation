import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


#------------------------------MultiLayerPerceptron--------------------------------------------#

activation_functions = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(),
                        'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation = 'relu',classification = True,bias = True):
        super().__init__()
        self.name = 'MultiLayerPerceptron'
        self.abbrv = 'mlp'
        self.classification = classification
        self.bias = bias
        # Construct layers
        model_layers = []
        previous_layer = input_dim
        for layer in hidden_layers:
            model_layers.append(nn.Linear(previous_layer, layer, bias=self.bias))
            model_layers.append(activation_functions[activation])
            previous_layer = layer
        n_class = 2 if classification else 1    
        model_layers.append(nn.Linear(previous_layer, n_class))
        self.network = nn.Sequential(*model_layers)
    
    def predict_layer(self, x, hidden_layer_idx=0, post_act=True):
        if hidden_layer_idx >= len(self.network) // 2:
            raise ValueError(f'The model has only {len(self.network) // 2} hidden layers, but hidden layer {hidden_layer_idx} was requested (indexing starts at 0).')
        
        network_idx = 2 * hidden_layer_idx + int(post_act)
        return self.network[:network_idx+1](x)
    
    def forward(self, x):
        if self.classification:
            y=F.softmax(self.network(x), dim=-1)
        else:
            y= self.network(x)
        return y
    
    def predict_with_logits(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        # Currently used by SHAP
        input = x if torch.is_tensor(x) else torch.from_numpy(np.array(x))
        return self.forward(input.float()).detach().numpy()
    
    def predict(self, x, argmax=False):
        # Currently used by LIME
        input = torch.squeeze(x) if torch.is_tensor(x) else torch.from_numpy(np.array(x))
        output = self.forward(input.float()).detach().numpy()
        return output.argmax(axis=-1) if argmax else output
    
#------------------------------MultiLayerPerceptron--------------------------------------------#   



#------------------------------FlowModel--------------------------------------------# 
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.flows import Flow, UnconditionalDistribution, NSF, UnconditionalTransform
from zuko.distributions import BoxUniform, DiagNormal
from zuko.transforms import SigmoidTransform

def flow(num_features, adjacency,hidden_features=[64,64]):

    
    flow_=  Flow(
        transform=[MaskedAutoregressiveTransform(features=num_features, 
                                        hidden_features=hidden_features,adjacency = adjacency,
                                        activation =torch.nn.LeakyReLU
                                    ),
                   UnconditionalTransform(SigmoidTransform),
        ], 
        base = UnconditionalDistribution(
        #            BoxUniform,
        #            torch.zeros(num_features),
        #            torch.ones(num_features)
                #    buffer=True,) if base_ == 'uniform' else 
                    DiagNormal, ##normal base
                    torch.zeros(num_features),
                    torch.ones(num_features),
                    buffer=True
                ) 
    
    )
    return flow_

#------------------------------FlowModel--------------------------------------------# 


#------------------------------AugmentedSCM--------------------------------------------# 
class AugmentedSCM(nn.Module):
    def __init__(self,scm_, mlp_):
        super(AugmentedSCM, self).__init__()
        self.mlp = mlp_
        self.device = next(self.mlp.parameters()).device
        self.scm = scm_.to(self.device)
        self.scm.eval()
        for param in self.mlp.parameters():
            param.requires_grad = False
        

    def forward(self,epsilon):
        model = self.scm()
        X = model.transform(epsilon)
        y= self.mlp(X)
        return y[:,0] 

#------------------------------AugmentedSCM--------------------------------------------# 


if __name__ == "__main__":
    from architectures import MLP,flow
    from utils import CONFIG
    config = CONFIG['german']
    # flow_ = flow(config['num_features'],config['gd_adjacency'])
    # print(flow_)
    Augscm = AugmentedSCM(flow(config['num_features'],config['gd_adjacency']),MLP(config['num_features'],config['hidden_layers_ann']))
    print(Augscm(torch.randn(10,config['num_features'])))
