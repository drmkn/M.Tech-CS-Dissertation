import torch
import json
import numpy as np
import ot
PREFIX = '/home/saptarshi/Dhruv/'
# PREFIX = '/user1/student/mtc/mtc2023/cs2306/Dhruv/Code/'
CONFIG = {'syn' : {'seed' : 10,'train_samples' : 2000, 'test_samples' : 600,'num_features' : 3,
                   'name' : 'syn', 'train_data' : PREFIX + 'Dissertation/datasets/synthetic_dataset/syn-train.csv',
                   'test_data' : PREFIX + 'Dissertation/datasets/synthetic_dataset/syn-test.csv',
                   'dag_path' : PREFIX + 'Dissertation/dags_estimated/syn_dag.json',
                   'graph_path' : PREFIX +'Dissertation/assets/est_syn_dag.png',
                   'target' : ['Y'], 'var_names' : ['W','Z','X'], 
                   'w_threshold' : 0.1,'lambda2' : 0.001, 'lambda1' : 0.01,
                   'gd_adjacency' : torch.tensor([[1,1,1],
                                                 [0,1,1],
                                                 [0,0,1]])
                   },

          'mpg' : {'seed' : 10,'train_samples' : 274, 'test_samples' : 118,'num_features' : 5,
                   'name' : 'mpg', 'train_data' : PREFIX + 'Dissertation/datasets/auto+mpg/mpg-train.csv',
                   'test_data' : PREFIX + 'Dissertation/datasets/auto+mpg/mpg-test.csv',
                   'dag_path' : PREFIX + 'Dissertation/dags_estimated/mpg_dag.json',
                   'graph_path' : PREFIX + 'Dissertation/assets/est_mpg_dag.png',
                   'target' : ['M'], 'var_names' : ['C','D','H','W','A'], 'w_threshold' : 0,
                   'lambda2' : 0.0, 'lambda1' : 0,
                   'gd_adjacency' : torch.tensor([[1,1,0,1,0],
                                                [0,1,1,0,1],
                                                [0,0,1,0,1],
                                                [0,0,0,1,1],
                                                [0,0,0,0,1]])
                   },
          'german' : {'seed' : 20,'train_samples' : 800, 'test_samples' : 200,'num_features' : 4,
                   'name' : 'german', 'train_data' : PREFIX +'Dissertation/datasets/german_credit/german-train.csv',
                   'test_data' : PREFIX + 'Dissertation/datasets/german_credit/german-test.csv',
                   'dag_path' : 'None',
                   'graph_path' : 'None',
                   'target' : ['R'], 'var_names' : ['G','A','C','D'], 'discrete_cols' : ['G','A','C','D'],
                   'batch_size' : 64,'hidden_features_flow' : [256,256],
                   'gd_adjacency' : torch.tensor([[1,0,1,0],
                                                  [0,1,1,0],
                                                  [0,0,1,1],
                                                  [0,0,0,1]])
                   }        
                   }


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


if __name__ == "__main__":
    config = CONFIG['syn']
    print(get_adjacency(config))
