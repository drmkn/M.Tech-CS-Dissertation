import torch
import json
import numpy as np
import pandas as pd
import ot
import networkx as nx
PREFIX = '/home/saptarshi/Dhruv/Dissertation/'
# PREFIX = '/user1/student/mtc/mtc2023/cs2306/Dhruv/Code/'
# PREFIX = '/home/dhruv/Files/Thesis/Dissertation/Code/'
CONFIG = {'syn' : {'seed' : 10,'train_samples' : 2000, 'test_samples' : 600,'num_features' : 3,
                   'name' : 'syn', 'train_data' : PREFIX + 'datasets/synthetic_dataset/syn-train.csv',
                   'test_data' : PREFIX + 'datasets/synthetic_dataset/syn-test.csv',
                   'dag_path' : PREFIX + 'dags_estimated/syn_dag.json',
                   'graph_path' : PREFIX +'assets/est_syn_dag.png',
                   'target' : ['S'], 'var_names' : ['P','Q','R'],'discrete_cols':[], 
                   'batch_size' : 64,'hidden_layers_flow' : [256,256],
                   'classification' : False, 'hidden_layers_mlp' : [256,256],
                   'w_threshold' : 0.1,'lambda2' : 0.001, 'lambda1' : 0.01,
                   'gd_adjacency' : torch.tensor([[1,1,1],
                                                 [0,1,1],
                                                 [0,0,1]]),
                    'meta_data' : ['c']*3,                            
                    'causal_graph' : nx.DiGraph([ (0,1),(0,2),
                                     (1,2)
                                    ]),
                    # 'exp_methods' : ["pfi","icc_topo","icc_shap"],                
                    'exp_methods' : ["shap","lime","sp_lime","pfi","ig","sg","itg","icc_topo","icc_shap"],
                    'features_names' : ['P','Q','R']                                             
                   },

          'mpg' : {'seed' : 10,'train_samples' : 274, 'test_samples' : 118,'num_features' : 5,
                   'name' : 'mpg', 'train_data' : PREFIX + 'datasets/auto+mpg/mpg-train.csv',
                   'test_data' : PREFIX + 'datasets/auto+mpg/mpg-test.csv',
                   'dag_path' : PREFIX + 'dags_estimated/mpg_dag.json',
                   'graph_path' : PREFIX + 'assets/est_mpg_dag.png',
                   'target' : ['M'], 'var_names' : ['C','D','H','W','A'], 'w_threshold' : 0,
                   'lambda2' : 0.0, 'lambda1' : 0,'batch_size' : 64,'hidden_layers_flow' : [256,256],
                   'classification' : False, 'hidden_layers_mlp' : [256,256],
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
                    'exp_methods' : ["shap","lime","sp_lime","pfi","ig","sg","itg","icc_topo","icc_shap"],
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
                    # 'exp_methods' : ["icc_shap"],                       
                    'exp_methods' : ["shap","lime","sp_lime","pfi","ig","sg","itg","icc_topo","icc_shap"],
                    'features_names' : ['gender','age','credit amount','repayment duration']                       
                   },
            'cancer' : {'seed' : 10,'train_samples' : 2400, 'test_samples' : 600,'num_features' : 7,
                'name' : 'cancer', 'train_data' : PREFIX +'datasets/lung_cancer/cancer-train.csv',
                'test_data' : PREFIX + 'datasets/lung_cancer/cancer-test.csv',
                'dag_path' : 'None','graph_path' : 'None','target' : ['D'], 
                'var_names' : ['A','T','S','L','B','E','X'], 
                'discrete_cols' : ['A','T','S','L','B','E','X'],'batch_size' : 64,'hidden_layers_flow' : [256,256],
                'classification' : True, 'hidden_layers_mlp' : [256,256],
                'meta_data' : ['c']*7,
                'gd_adjacency' : torch.tensor([ [1,1,0,0,0,0,0],
                                                [0,1,0,0,0,1,0],
                                                [0,0,1,1,1,0,0],
                                                [0,0,0,1,0,1,0],
                                                [0,0,0,0,1,0,0],
                                                [0,0,0,0,0,1,1],
                                                [0,0,0,0,0,0,1]]),
                'causal_graph' : nx.DiGraph([(0,1),(1,5),(2,3),(2,4),
                                             (3,5),(5,6)]),
                # 'exp_methods' : ["icc_topo"],                       
                'exp_methods' : ["shap","lime","sp_lime","pfi","ig","sg","itg","icc_topo","icc_shap"],
                'features_names' : ['asia','tub','smoke','lung','bronc','either','xray']                       
                }        
                   }

#for "ig","itg","sg","shap" local explanations
openxai_config ={"ig": {
            "method": "gausslegendre", 
            "multiply_by_inputs": False},
                "itg": {},
                "sg": {
                "n_samples": 500,
                "standard_deviation": 0.1,
                "seed": 1
            },
            "shap": {
                "n_samples": 500,
                "model_impl": "torch",
                "seed": 1
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

import lime
import lime.lime_tabular
from lime import submodular_pick
def sp_lime(data,features_names,class_names, network, task='classification' ):
    num_features= data.size(1)
    data = data.detach().numpy()
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=data,
                                                       feature_names= features_names,
                                                        mode=task,random_state=1,
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
                                                        mode=task,random_state=1,
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

import networkx as nx
import os
import matplotlib.pyplot as plt
def generate_causal_graph(config):
    num_vars = config['num_features']
    dag = config['gd_adjacency'] - torch.eye(num_vars) # 1's on the diagonal were required for the flow model
    G = nx.DiGraph()
    G.add_nodes_from(range(num_vars))

    # Add edges from adjacency matrix
    for i in range(num_vars):
        for j in range(num_vars):
            if abs(dag[i, j]) > 0:  # Non-zero edge
                G.add_edge(i, j)

    # Map variable names to nodes if available
    if 'var_names' in config:
        mapping = {i: name for i, name in enumerate(config['var_names'])}
        G = nx.relabel_nodes(G, mapping)

    # Save figure
    os.makedirs("assets", exist_ok=True)
    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G,seed=config['seed'])
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_weight='bold', arrows=True)
    plt.title(f"Ground Truth Causal DAG assumed for {config['name']} dataset", pad=20)

    # Adjust layout and save with bbox_inches
    plt.tight_layout()
    plt.savefig(f"assets/{config['name']}_dag.png", bbox_inches='tight')
    plt.close()


##########################Explanations evaluation#######################################################
import torch
import pandas as pd
from OpenXAI.openxai.explainers.perturbation_methods import BasePerturbation
from OpenXAI.openxai.experiment_utils import generate_mask

class Perturbation(BasePerturbation):
    def __init__(self, data_format):
        super(Perturbation, self).__init__(data_format)

    def get_perturbed_inputs(self, original_sample: torch.FloatTensor, feature_mask: torch.BoolTensor,
                             num_samples: int) -> torch.Tensor:
        """
        Generates num_samples perturbations of a single input,
        by applying Gaussian noise to features where feature_mask is False.
        """
        assert len(feature_mask) == len(original_sample), \
            f"Feature mask must match original sample shape in {self.__class__}"
        
        perturbations = torch.randn(num_samples, len(original_sample))  # [num_samples, F]
        perturbed_samples = original_sample + perturbations * (~feature_mask)  # noise on unmasked features
        
        return perturbed_samples  # shape: [num_samples, F]


@torch.no_grad()
def pred_faith(k, inputs, explanations, invert, model, perturb_method: Perturbation):
    """
    Computes PGI or PGU depending on 'invert'.
    """
    top_k_mask = generate_mask(explanations, k)            # [N, F]
    top_k_mask = ~top_k_mask if invert else top_k_mask

    metric_per_sample = []

    for i in range(inputs.shape[0]):
        x = inputs[i]           # shape: [F]
        mask = top_k_mask    # shape: [F]

        if x.ndim == 0 or mask.ndim == 0:
            raise ValueError(f"Input or mask is 0-D: x.shape = {x.shape}, mask.shape = {mask.shape}")

        x_perturb = perturb_method.get_perturbed_inputs(
            original_sample = x,
            feature_mask    = mask,
            num_samples     = 1000
        )  # shape: [1000, F]

        y = model(x.unsqueeze(0))             # shape: [1, D]
        y_perturb = model(x_perturb)          # shape: [1000, D]

        gap = torch.mean(torch.abs(y_perturb[:, 0] - y[0, 0]))  # scalar
        metric_per_sample.append(gap.item())

    return torch.tensor(metric_per_sample)
    # return torch.tensor(metric_per_sample).mean().item(), torch.tensor(metric_per_sample).std().item()

def evaluate_exp(ge_dict, config, mlp_model):
    n = config['num_features']
    evaluation_metrics = dict()
    evaluation_metrics_per_sample = dict()
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Load test data
    df = pd.read_csv(config['test_data'])
    inputs = torch.tensor(df.drop(columns=config['target']).values).float()  # [N, F]

    for method, features in ge_dict.items():
        evaluation_metrics[method] = {'pgi': {}, 'pgu': {}}
        evaluation_metrics_per_sample[method] = {'pgi': {}, 'pgu': {}}
        # Stack explanations into tensor [N, F]
        exp = [attr[0] for _, attr in features.items()]
        ge = torch.tensor(exp).float().to('cpu')  # [N, F]

        perturb_method = Perturbation("tabular")

        for k in range(1, n + 1):
            pgi = pred_faith(
                k=k,
                inputs=inputs,
                explanations=ge,
                invert=False,  # PGI: perturb top-k
                model=mlp_model,
                perturb_method=perturb_method
            )
            pgu = pred_faith(
                k=k,
                inputs=inputs,
                explanations=ge,
                invert=True,   # PGU: perturb all-but-top-k
                model=mlp_model,
                perturb_method=perturb_method
            )

            evaluation_metrics[method]['pgi'][f'k={k}'] = [pgi.mean().item()]
            evaluation_metrics[method]['pgu'][f'k={k}'] = [pgu.mean().item()]
            evaluation_metrics_per_sample[method]['pgi'][f'k={k}'] = pgi
            evaluation_metrics_per_sample[method]['pgu'][f'k={k}'] = pgu

    return evaluation_metrics,evaluation_metrics_per_sample


def aggregate_auc_or_sum(pgx_dict,config):
    auc_results = {}
    auc_show = {}
    sum_results = {}
    sum_show = {}
    ks = range(1,config['num_features']+1)

    for method, metrics in pgx_dict.items():
        auc_results[method] = {}
        auc_show[method] = {}
        sum_results[method] = {}
        sum_show[method] = {}


        for metric_type in ['pgi', 'pgu']:
            auc_results[method][f'{metric_type}_auc_per_sample'] = []
            sum_results[method][f'{metric_type}_sum_per_sample'] = []
            for i in range(config['test_samples']):
                y_vals = [metrics[metric_type][f'k={k}'][i] for k in ks]
                auc = np.trapz(y_vals, ks)/ (ks[-1] - ks[0])  # normalize
                auc_results[method][f'{metric_type}_auc_per_sample'].append(auc)
                sum_results[method][f'{metric_type}_sum_per_sample'].append(sum(y_vals))
            auc_mean,auc_std_err = torch.tensor(auc_results[method][f'{metric_type}_auc_per_sample']).mean().item(),torch.tensor(auc_results[method][f'{metric_type}_auc_per_sample']).std().item()/(config['test_samples'])**0.5    
            auc_show[method][f"{metric_type}_auc"] = f"{auc_mean} +- {auc_std_err}"
            sum_mean,sum_std_err = torch.tensor(sum_results[method][f'{metric_type}_sum_per_sample']).mean().item(),torch.tensor(sum_results[method][f'{metric_type}_sum_per_sample']).std().item()/(config['test_samples'])**0.5    
            sum_show[method][f"{metric_type}_sum"] = f"{sum_mean} +- {sum_std_err}"

    return auc_results, auc_show,sum_results, sum_show

# import json
# from typing import Dict

# def compute_pgu_pgi_sums(evaluation_metrics: Dict,config) -> Dict[str, Dict[str, float]]:
#     """
#     Compute the sum of PGU and PGI values across all k for each attribution method.

#     Parameters:
#     - evaluation_metrics (dict): Dictionary containing PGU and PGI values for each method.

#     Returns:
#     - dict: Dictionary with method names as keys and their PGU/PGI sums as values.
#     """
#     sum_results = {}
#     sum_show = {}
#     ks = range(1,config['num_features']+1)

#     for method, metrics in evaluation_metrics.items():
#         sum_results[method] = {}
#         sum_show[method] = {}

#         for metric_type in ['pgi', 'pgu']:
#             auc_results[method][f'{metric_type}_auc_per_sample'] = []
#             for i in range(config['test_samples']):
#                 y_vals = [metrics[metric_type][f'k={k}'][i] for k in ks]
#                 auc = np.trapz(y_vals, ks)/ (ks[-1] - ks[0])  # normalize
#                 auc_results[method][f'{metric_type}_auc_per_sample'].append(auc)
#             mean,std_err = torch.tensor(auc_results[method][f'{metric_type}_auc_per_sample']).mean().item(),torch.tensor(auc_results[method][f'{metric_type}_auc_per_sample']).std().item()/(config['test_samples'])**0.5    
#             auc_show[method][f"{metric_type}_auc"] = f"{mean} +- {std_err}"

#     return auc_results, auc_show





    





if __name__ == "__main__":
    # config = CONFIG['syn']
    # print(get_adjacency(config))
    P = Perturbation("tabular")
    print(P.get_perturbed_inputs(torch.tensor([[1,2,3,4],[5,6,7,8]]),torch.tensor([[True, False, False,False],[False, True, False,False]], dtype=torch.bool),
                                 10,['c']*4))

