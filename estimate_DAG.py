'''Estimate DAG of a given dataset using notears algorithm'''
import torch
import sys
import os 
sys.path.append('/home/saptarshi/Dhruv/Dissertation/notears')
from notears.nonlinear import notears_nonlinear,NotearsMLP
import pandas as pd
import numpy as np
from utils import CONFIG
import networkx as nx
import matplotlib.pyplot as plt
import json

name = 'mpg'
config = CONFIG[name]
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.set_default_dtype(torch.double)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv(filepath_or_buffer=config['train_data'])
data.drop(columns=config['to_drop'],inplace=True)
X = torch.tensor(data.to_numpy(), dtype=torch.double).to(device)

model = NotearsMLP(dims=[data.shape[1],10, 1]).to(device)

dag = notears_nonlinear(model=model, X=X.cpu().numpy(),lambda1=config.get('lambda1',0)
                        ,lambda2=config.get('lambda2',0),w_threshold=config.get('w_threshold',0))  # notears_nonlinear still expects numpy
print(dag)

# Ensure folder exists
os.makedirs("dags_estimated", exist_ok=True)

# Convert NumPy array to a native list
dag_list = dag.tolist()

# Save to JSON file in the folder
with open(config['dag_path'], "w") as f:
    json.dump(dag_list, f, indent=4)


# Create and save DAG graph
G = nx.DiGraph()

# Add nodes
num_vars = dag.shape[0]
G.add_nodes_from(range(num_vars))

# Add edges from adjacency matrix
for i in range(num_vars):
    for j in range(num_vars):
        if abs(dag[i, j]) > 1e-6:  # Non-zero edge
            G.add_edge(i, j)

# Map variable names to nodes if available
if 'var_names' in config:
    mapping = {i: name for i, name in enumerate(config['var_names'])}
    G = nx.relabel_nodes(G, mapping)

# Save figure
os.makedirs("assets", exist_ok=True)
plt.figure(figsize=(6, 5))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_weight='bold', arrows=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
plt.title("Estimated Causal DAG")
# plt.tight_layout()
plt.savefig(config['graph_path'])
plt.close()

