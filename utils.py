import torch
# PREFIX = '/home/saptarshi/Dhruv/'
PREFIX = '/user1/student/mtc/mtc2023/cs2306/Dhruv/Code/'
CONFIG = {'syn' : {'seed' : 10,'train_samples' : 2000, 'test_samples' : 600,
                   'name' : 'syn', 'train_data' : PREFIX + 'Dissertation/datasets/synthetic_dataset/syn-train.csv',
                   'test_data' : PREFIX + 'Dissertation/datasets/synthetic_dataset/syn-test.csv',
                   'dag_path' : PREFIX + 'Dissertation/dags_estimated/syn_dag.json',
                   'graph_path' : PREFIX +'Dissertation/assets/est_syn_dag.png',
                   'target' : ['Y'], 'var_names' : ['W','Z','X'], 
                   'w_threshold' : 0.1,'lambda2' : 0.001, 'lambda1' : 0.01
                   },

          'mpg' : {'seed' : 10,'train_samples' : 274, 'test_samples' : 118,
                   'name' : 'syn', 'train_data' : PREFIX + 'Dissertation/datasets/auto+mpg/mpg-train.csv',
                   'test_data' : PREFIX + 'Dissertation/datasets/auto+mpg/mpg-test.csv',
                   'dag_path' : PREFIX + 'Dissertation/dags_estimated/mpg_dag.json',
                   'graph_path' : PREFIX + 'Dissertation/assets/est_mpg_dag.png',
                   'target' : ['M'], 'var_names' : ['C','D','H','W','A'], 'w_threshold' : 0.3,
                   'lambda2' : 0.01, 'lambda1' : 0.1
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