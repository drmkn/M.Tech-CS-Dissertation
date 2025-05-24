import torch
import json
import numpy as np
import pandas as pd
import ot
import networkx as nx
# PREFIX = '/home/saptarshi/Dhruv/Dissertation/'
# PREFIX = '/user1/student/mtc/mtc2023/cs2306/Dhruv/Code/'
PREFIX = '/home/dhruv/Files/Thesis/Dissertation/Code/'
CONFIG = {'syn' : {'seed' : 10,'train_samples' : 2000, 'test_samples' : 600,'num_features' : 3,
                   'name' : 'syn', 'train_data' : PREFIX + 'datasets/synthetic_dataset/syn-train.csv',
                   'test_data' : PREFIX + 'datasets/synthetic_dataset/syn-test.csv',
                   'dag_path' : PREFIX + 'dags_estimated/syn_dag.json',
                   'graph_path' : PREFIX +'assets/est_syn_dag.png',
                   'target' : ['Y'], 'var_names' : ['W','Z','X'], 
                   'batch_size' : 64,'hidden_layers_flow' : [256,256],
                   'classification' : False, 'hidden_layers_mlp' : [100,100],
                   'w_threshold' : 0.1,'lambda2' : 0.001, 'lambda1' : 0.01,
                   'gd_adjacency' : torch.tensor([[1,1,1],
                                                 [0,1,1],
                                                 [0,0,1]]),
                    'meta_data' : ['c']*3,                            
                    'causal_graph' : nx.DiGraph([ (0,1),(0,2),
                                     (1,2)
                                    ]),
                    # 'exp_methods' : ["pfi","icc_topo","icc_shap"],                
                    'exp_methods' : ["ig","itg","sg","shap","lime","sp_lime","pfi","icc_topo","icc_shap"],
                    'features_names' : ['W','Z','X']                                             
                   },

          'mpg' : {'seed' : 10,'train_samples' : 274, 'test_samples' : 118,'num_features' : 5,
                   'name' : 'mpg', 'train_data' : PREFIX + 'datasets/auto+mpg/mpg-train.csv',
                   'test_data' : PREFIX + 'datasets/auto+mpg/mpg-test.csv',
                   'dag_path' : PREFIX + 'dags_estimated/mpg_dag.json',
                   'graph_path' : PREFIX + 'assets/est_mpg_dag.png',
                   'target' : ['M'], 'var_names' : ['C','D','H','W','A'], 'w_threshold' : 0,
                   'lambda2' : 0.0, 'lambda1' : 0,'batch_size' : 64,'hidden_layers_flow' : [256,256],
                   'classification' : False, 'hidden_layers_mlp' : [100,100],
                   'gd_adjacency' : torch.tensor([[1,1,0,1,0],
                                                [0,1,1,0,1],
                                                [0,0,1,0,1],
                                                [0,0,0,1,1],
                                                [0,0,0,0,1]]),
                    'meta_data' : ['c']*5,                            
                    'causal_graph' : nx.DiGraph([ (0,1),(0,3),
                                           (1,2),(1,4),
                                           (2,4),
                                           (3,4)]),
                    'exp_methods' : ["ig","itg","sg","shap","lime","sp_lime","pfi","icc_topo","icc_shap"],
                    'features_names' : ['cylinders','displacement','horsepower','weight','acceleration']                                                  
                   },
          'german' : {'seed' : 30,'train_samples' : 800, 'test_samples' : 200,'num_features' : 4,
                   'name' : 'german', 'train_data' : PREFIX +'datasets/german_credit/german-train.csv',
                   'test_data' : PREFIX + 'datasets/german_credit/german-test.csv',
                   'dag_path' : 'None','graph_path' : 'None','target' : ['R'], 'var_names' : ['G','A','C','D'], 
                   'discrete_cols' : ['G','A','C','D'],'batch_size' : 64,'hidden_layers_flow' : [256,256],
                   'classification' : True, 'hidden_layers_mlp' : [256,256],
                   'meta_data' : ['c']*4,
                   'gd_adjacency' : torch.tensor([[1,0,1,0],
                                                  [0,1,1,0],
                                                  [0,0,1,1],
                                                  [0,0,0,1]]),
                    'causal_graph' : nx.DiGraph([ (0,2),(1,2),
                                           (1,2),(2,3)]),
                    'exp_methods' : ["sp_lime","pfi","icc_topo","icc_shap"],
                    'features_names' : ['gender','age','credit amount','repayment duration']                       
                   }        
                   }

#for "ig","itg","sg","shap" local explanations
openxai_config ={"ig": {
            "method": "gausslegendre", 
            "multiply_by_inputs": False},
                "itg": {},
                "sg": {
                "n_samples": 100,
                "standard_deviation": 0.1,
                "seed": 0
            },
            "shap": {
                "n_samples": 500,
                "model_impl": "torch",
                "seed": 0
            }}


def MMD(x, y, lengthscale):
    """ Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    """
    
    # Check if x and y are 1D arrays and reshape them to 2D if necessary
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

    m = x.shape[0]
    n = y.shape[0]

    # Concatenate x and y along the 0th axis
    z = torch.cat((x, y), dim=0)

    # Compute the kernel matrix
    K = kernel_matrix(z, z, lengthscale)

    # Extract submatrices for x, y, and their interaction
    kxx = K[:m, :m]
    kyy = K[m:, m:]
    kxy = K[:m, m:]

    # Compute the MMD
    return (1 / m ** 2) * torch.sum(kxx) - (2 / (m * n)) * torch.sum(kxy) + (1 / n ** 2) * torch.sum(kyy)


# Kernel matrix computation
def kernel_matrix(x, y, l):
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
    return torch.exp(-(1 / (2 * l ** 2)) * torch.cdist(x, y, p=2).pow(2))


def get_adjacency(config):
    with open(config['dag_path'], 'r') as f:
        dag = json.load(f)
    
    dag_adj = np.array(dag)

    binary_adj = (dag_adj != 0) + np.identity(dag_adj.shape[0])

    return torch.tensor(binary_adj,dtype=torch.float32)

def wasserstein_distance(sample_1,sample_2,p=1,return_tensor=False):
    if sample_1.device != 'cpu':
        sample_1 = sample_1.to('cpu')
    if sample_2.device != 'cpu':
        sample_2 = sample_2.to('cpu')

    assert sample_1.size(0) == sample_2.size(0)
    num_sample = sample_1.size(0)
    cost_matrix = ot.dist(x1=sample_1.numpy(),x2=sample_2.numpy(),metric='minkowski',p=p,w=None)
    distance =ot.emd2(a=np.ones(num_sample)/num_sample,
                      b=np.ones(num_sample)/num_sample,
                      M=cost_matrix,
                      numItermax=1000000,
                      )
    # ot_matrix = ot.emd(a=np.ones(num_sample)/num_sample,
    #                   b=np.ones(num_sample)/num_sample,
    #                   M=cost_matrix)
    # print(ot_matrix)
    if return_tensor:
        distance = torch.tensor(distance)
    return distance

def log_adjacency_as_text(logger, adj, var_names, tag="Adjacency Matrix"):
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    adj = np.array(adj)

    # Header row
    header = "       " + "  ".join(f"{name:>6}" for name in var_names) + "\n"

    # Each row
    lines = []
    for i, row in enumerate(adj):
        row_str = "  ".join(f"{val:6.1f}" for val in row)
        lines.append(f"{var_names[i]:>6}  {row_str}")

    formatted_text = header + "\n".join(lines)
    logger.experiment.add_text(tag, f"```\n{formatted_text}\n```", global_step=0)

def convert_to_cont(df,discrete_cols,seed):
    '''required for the CNF model training'''
    np.random.seed(seed)
    for col in discrete_cols:
        df[col] = df[col] + np.random.rand(df[col].shape[0])
        
    return df

def torch_to_csv(tensor,name:str,header):
    pd.DataFrame(tensor.numpy()).to_csv(name,index=False, header=header)


def attr_to_dict(attributions):
    feature_names, attributions = zip(*attributions)
    feature_names = list(feature_names)
    attributions = list(attributions)
    n = len(feature_names)
    attr_dict = dict()
    for i in range(n):
        attr_dict[feature_names[i]] = [attributions[i]]

    return attr_dict    

def create_feature_attribution_output(feature_names, attribution):
    # Combine feature names and attribution values into a list of tuples
    feature_attribution = list(zip(feature_names, attribution.tolist()))
    
    # Sort the list of tuples by attribution values in descending order
    #feature_attribution.sort(key=lambda x: x[1], reverse=True)
    
    return feature_attribution

import torch
import pandas as pd
from OpenXAI.openxai.explainers.perturbation_methods import BasePerturbation
from OpenXAI.openxai.experiment_utils import generate_mask

class Perturbation(BasePerturbation):
    def __init__(self, data_format):
        super(Perturbation, self).__init__(data_format)

    def get_perturbed_inputs(self, original_sample: torch.FloatTensor, feature_mask: torch.BoolTensor,
                             num_samples: int, feature_metadata: list) -> torch.tensor:
        '''
        feature mask : this indicates the static features
        num_samples : number of perturbed samples.
        '''
        feature_type = feature_metadata
        assert len(feature_mask) == len(original_sample),\
            f"mask size == original sample in get_perturbed_inputs for {self.__class__}"
        
        
        # Processing continuous columns
        #torch.manual_seed(0)
        # perturbations =  torch.rand([num_samples, len(feature_type)]) 
        perturbations =  torch.randn([num_samples, len(feature_type)])
        # print(perturbations)

        
        # keeping features static that are in top-K based on feature mask
        perturbed_samples = original_sample * feature_mask  #+ perturbations * (~feature_mask)
        
        return perturbed_samples



def pred_faith(k, inputs, targets, task, explanations, invert, model,  perturb_method:Perturbation,
                           feature_metadata, ):#n_samples, seed):
    seeds = [10]
    top_k_mask =  generate_mask(explanations, k)
    top_k_mask = torch.logical_not(top_k_mask) if invert else top_k_mask
    #print(top_k_mask)

    metrics1=[]
    metrics2=[]
    for seed in seeds:
        torch.manual_seed(seed)
        x_perturb = perturb_method.get_perturbed_inputs(original_sample= inputs,
                                                    feature_mask=top_k_mask, 
                                                    num_samples=inputs.shape[0], feature_metadata=feature_metadata ) 
    #print(torch.abs(x_perturb-inputs)[0:10])
        y = model(inputs)
        y_perturb = model(x_perturb)
       #y - targets   ---> RMSE              if regression
       #y_perturb - targets  ---> RMSE

       #if classification
       #  y---> class label ---> accuracy
       # y_perturb ---> class label --->accuracy
        if task == "regression":
            rmse1 = torch.sqrt(torch.mean((y - targets) ** 2))
            rmse2 = torch.sqrt(torch.mean((y_perturb - targets) ** 2))
            metric2 = rmse2-rmse1
            metrics2.append(torch.tensor(metric2))
        elif task == "classification":
            accuracy1 = (targets == torch.argmax(y, dim=1)).sum().item() / targets.size(0)
            accuracy2 = (targets == torch.argmax(y_perturb, dim=1)).sum().item() / targets.size(0)
            metric2 = accuracy2-accuracy1
            metrics2.append(torch.tensor(metric2))
            

    
        metric1 = torch.mean(torch.abs(y-y_perturb)[:,0])
        metrics1.append(metric1)
    
    # print(metrics1,metrics2)
    return torch.mean(torch.stack(metrics1)),torch.mean(torch.stack(metrics2)) #metrics
    # return torch.tensor(metric)



import lime
import lime.lime_tabular
from lime import submodular_pick
def sp_lime(data,features_names,class_names, network, task='classification' ):
    num_features= data.size(1)
    data = data.detach().numpy()
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=data,
                                                       feature_names= features_names,
                                                        mode=task,random_state=0,
                                                        categorical_features=None,
                                                        verbose=False,
                                                        discretize_continuous=False,
                                                        class_names=class_names
    )

    predict_fn =lambda  data: network.forward(torch.from_numpy(data).float()).detach().numpy()
    sp_obj = submodular_pick.SubmodularPick(explainer= explainer,data=data, 
                                            predict_fn=predict_fn, method='full', num_exps_desired=1 ,
                                            num_features=num_features
    )

    # Attempt to retrieve explanations for class label 1, or fallback to 0 if not available
    explanation = sp_obj.sp_explanations[0]
    target_label = 1 if 1 in explanation.local_exp else 0

    # Print to debug which labels are available and which one is selected
    # print("Available labels in explanation:", list(explanation.local_exp.keys()))
    # print("Selected label:", target_label)

    return normalize_abs_sum(explanation.as_list(label=target_label))
    
    # print(sp_obj.sp_explanations)

    # return normalize_abs_sum(sp_obj.sp_explanations[0].as_list())
    # return sp_obj.sp_explanations[0].as_list()



def normalize_abs_sum(data, value_index=1):
    # Extract the numeric values from the data
    values = [x[value_index] for x in data]
    total_abs = sum(abs(v) for v in values)
    
    if total_abs == 0:
        normalized_values = [0 for _ in values]
    else:
        normalized_values = [abs(v) / total_abs for v in values]
    
    # Reconstruct the tuples with normalized values
    normalized_data = [
        x[:value_index] + (normalized_values[i],) + x[value_index+1:]
        for i, x in enumerate(data)
    ]
    
    return normalized_data

def generate_lime_exp(data,features_names,class_names, network, task='classification'):
    num_features= data.size(1)
    data = data.detach().numpy()
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=data,
                                                       feature_names= features_names,
                                                        mode=task,random_state=0,
                                                        categorical_features=None,
                                                        verbose=False,
                                                        discretize_continuous=False,
                                                        class_names=class_names
    )

    predict_fn =lambda  data: network.forward(torch.from_numpy(data).float()).detach().numpy()
    local_explanations = {}
    for feature in features_names:
        local_explanations[feature] = []
    for i in range(data.shape[0]):
        exp = explainer.explain_instance(
            data[i], 
            predict_fn, 
            num_features=num_features 
        )
        for feature,attr in exp.as_list():
            local_explanations[feature].append(attr)
    return local_explanations









if __name__ == "__main__":
    config = CONFIG['syn']
    print(get_adjacency(config))
