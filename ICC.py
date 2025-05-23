import torch
import torch.nn as nn 
import networkx as nx
from itertools import permutations
import itertools
import scipy.stats.qmc as qmc
import math

def all_topological_sorts(G: nx.DiGraph,dim:int):    
    if  len(G.edges()) == 0:
        assert dim is not None
        sort_ =list(permutations(range(dim)))
        sort = [list(p) for p in sort_]
    else:
        if len(G.nodes())!= dim:
            missing_nodes = set(range(dim)) - set(G.nodes())
            print(missing_nodes)
            G.add_nodes_from(missing_nodes)
        sort =list(nx.all_topological_sorts(G))
    return sort 

def keep_elements_before_B_in_A(A, B):
    # Find indices where A equals B
    if A.dim() == 0:
        A = A.unsqueeze(0)
    indices = (A == B).nonzero(as_tuple=False)
    if indices.numel() == 0 :
        # B is not in A; return A unchanged
        return A
    else:
        # Get the first index where A equals B
        idx = indices[0].item()
        # Return elements before B
        return A[:idx]


def Intrinsic_Causal_Contribution(neural_network, topological_orederings, dim,  rqmc=False, sample_size=512  ):
    ICC_=[]
    for i in range(dim):
        ICC_.append(
            ICC(variable =torch.tensor([i]),
                model= neural_network,
                topological_orderings=topological_orederings, #all_topological_sorts(graph,dim),
                dim=dim,
                rqmc=rqmc, num_samples= sample_size).detach().unsqueeze(0)
        )
    return ICC_

def ICC(variable,model, topological_orderings: list, dim:int,
        rqmc:bool =False, 
        num_samples: int =512,  seed: int=0  ):
    #print(topological_orderings,len(topological_orderings))
    if not isinstance(variable, torch.Tensor):
        variable = torch.tensor([variable])
    assert isinstance(model, nn.Module), "The model is not a subclass of nn.Module"

    shape = (num_samples,dim)
    if not rqmc:
        torch.manual_seed(0)
        sample_U = torch.randn(shape,dtype=torch.float32,device=model.device)
        sample_V = torch.randn(shape,dtype=torch.float32,device=model.device)
    else:
        #torch.manual_seed(0)
        engine = torch.quasirandom.SobolEngine(dimension=dim,scramble=True)
        sample_U = engine.draw(num_samples,dtype=torch.float32 ).to(model.device)
        sample_V = engine.draw(num_samples,dtype=torch.float32,).to(model.device)
        #engine = qmc.Sobol(d=dim,scramble=True)
        #sample_U = torch.from_numpy( engine.random(num_samples) ).float()
        #sample_U = torch.rand(shape,dtype=torch.float32)
        #sample_V = torch.from_numpy( engine.random(num_samples) ).float()
    y_u =   model.forward(sample_U)   
    y_v =   model.forward(sample_V) 
    #print("U",y_u.mean())
    #print("V",y_v.mean())

    icc=0.0
    ic=[]
    for ordering in topological_orderings:
        
        context=keep_elements_before_B_in_A(torch.tensor(ordering), variable)
        #print(ordering,context)
        icc_context = ICC_context(variable=variable, context=context, trained_model=model, y_U=y_u, y_V=y_v,sample_U=sample_U,sample_V=sample_V ,dim=dim )
        icc += icc_context
        ic.append(icc_context)
    #print({variable:ic})

    return  icc/ len(topological_orderings)



def ICC_SHAP(dim, model, sample_size=512, rqmc=False):
    assert isinstance(model, nn.Module), "The model is not a subclass of nn.Module"
    ICC_ =[]
    T = torch.tensor([range(dim)])
    for i in range(dim):
        variable =torch.tensor([i])
        shape = (sample_size,dim)
        if not rqmc:
            torch.manual_seed(0)
            sample_U = torch.randn(shape,dtype=torch.float32,device=model.device)
            sample_V = torch.randn(shape,dtype=torch.float32,device=model.device)
        else:
        #torch.manual_seed(0)
            engine = torch.quasirandom.SobolEngine(dimension=dim,scramble=True)
            sample_U = engine.draw(sample_size,dtype=torch.float32 ).to(model.device)
            sample_V = engine.draw(sample_size,dtype=torch.float32,).to(model.device)
        y_u =   model.forward(sample_U)   
        y_v =   model.forward(sample_V) 
        icc =0.0
        for subset in find_subsets(T[T!=variable.item()]):
            context = subset 
            #print(variable,context,subset.shape[0]) 
            icc_context = ICC_context(variable=variable, context=context, 
                                      trained_model=model, 
                                      y_U=y_u, y_V=y_v,sample_U=sample_U,sample_V=sample_V ,
                                      dim=dim )
            icc +=  1/ (dim* math.comb(dim-1,subset.shape[0])) * icc_context
        ICC_.append(icc.detach().unsqueeze(0))

    return ICC_
            
def ICC_context(variable,context,trained_model, y_U, y_V, sample_U, sample_V, dim:int):
    assert isinstance(trained_model, nn.Module), "The model is not a subclass of nn.Module"  
    num_samples = sample_U.size(0)
    dim = sample_U.size(1)

    I = context 
    neg_I =  torch.arange(dim)[~torch.isin(torch.arange(dim), I)]
    sample_W1 = torch.empty(num_samples,dim,device=trained_model.device)
    sample_W1[:,neg_I] = sample_U[:,neg_I]
    sample_W1[:,I] = sample_V[:,I]
    y_W1 =   trained_model.forward(sample_W1) 

    # if I.numel() == 0:
    #     print('numel=0', torch.var(y_U)- 0.5* torch.mean((y_V-y_W1)**2),    
    #           torch.var(torch.cat((y_U,y_V),dim=0)) - 0.5* torch.mean((y_V-y_W1)**2)
    #           ) 


    J= torch.cat((context, variable))
    neg_J =  torch.arange(dim)[~torch.isin(torch.arange(dim), J)]
    sample_W2 = torch.empty(num_samples,dim,device = trained_model.device)
    sample_W2[:,neg_J] = sample_U[:,neg_J]
    sample_W2[:,J] = sample_V[:,J]
    y_W2 =   trained_model.forward(sample_W2) 

    #print('W1W2',y_W1.mean(),y_W2.mean())
    #print( variable, I,J)
    var_context_variable, var_y_1 = var_expectation(y_U,y_V,y_W2)
    var_context, var_y_2 = var_expectation(y_U,y_V,y_W1)
    assert var_y_1 == var_y_2
    var_y = var_y_1
    #if I.numel() == 0:
        #print('numel=0',var_context)

    if J.numel() == dim:
        assert var_context_variable== var_y
    #print('var',var_y)
    diff = max(var_context_variable - var_context , 1e-12)
    #print('diff',var_context_variable, var_context, diff )
    return diff / var_y #phi_var_context - phi_context

def var_expectation(y_U,y_V, y_W ):
    var_y =  torch.var(torch.concat([y_U,y_V],dim=0)) #(torch.var(y_U) + torch.var(y_V))/2
    phi =max( var_y - 0.5* torch.mean((y_V-y_W)**2) ,1e-12) # for numerical stability  
    return phi, var_y

def find_subsets(input_tensor):
    input_list = input_tensor.tolist()  # Convert tensor to list
    subsets = []
    for r in range(len(input_list) + 1):
        subsets.extend(itertools.combinations(input_list, r))
    # Convert each subset back to a tensor
    tensor_subsets = [torch.tensor(subset).int() for subset in subsets]
    return tensor_subsets


