import numpy as np
import pandas as pd
import torch,os
from utils import generate_lime_exp,sp_lime,attr_to_dict,create_feature_attribution_output,openxai_config
from architectures import AugmentedSCM,MLP,flow
from pl_modules import MLPLightning,CausalNF
from OpenXAI.openxai import Explainer
from OpenXAI.openxai.experiment_utils import fill_param_dict
from gam.gam import GAM
from captum.attr import FeaturePermutation
from ICC import Intrinsic_Causal_Contribution, all_topological_sorts,ICC_SHAP
import plotly.graph_objects as go
import plotly.graph_objects as go
import json
from utils import Perturbation, pred_faith,CONFIG,generate_causal_graph
import time
import plotly.io as pio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_indices = {0:"ig", 1:"itg", 2:"sg", 3:"shap", 4:"lime", 5:"sp_lime", 6:"pfi", 7:"icc_topo", 8:"icc_shap"}
methods_type = {"ig":'local',"itg":'local',"sg":'local',"shap":'local',"lime":'local',"sp_lime":'global',"pfi":'global',"icc_topo":'global',"icc_shap":'global'}
method_names = {"ig":'IG',"itg":'IxG',"sg":'SG',"shap":'SHAP',"lime":'LIME',"sp_lime":'SP LIME',"pfi":'PFI',"icc_topo":'ICC TOPO',"icc_shap":'ICC SHAP'}
def generate_global_exps(config, mlp_model, scm_model):
    df = pd.read_csv(config['test_data'])
    inputs = torch.tensor(df.drop(columns=config['target']).values)
    targets = torch.tensor(df[config['target']].values,dtype=torch.long).squeeze()
    if config['classification']:
        task = 'classification'
        class_names = [0,1]
        labels = targets
    else:
        task = 'regression'
        class_names = None 
        labels = None 
    # data_name = config['name']    
    methods = config['exp_methods']
    features_names = config['features_names']
    m = len(methods)
    n = len(features_names)
    evaluation_metrics = dict()
    time_dict = dict()
    ge_dict = dict()
    global_explanations = dict()

    for feature_name in features_names:
        global_explanations[feature_name] = []
    for method in methods:
        evaluation_metrics[method] = dict() 
        time_dict[method] = 0
        ge_dict[method] = dict()

    # rqmcs = [False]
    for method in methods:
        # for rqmc in rqmcs:
            if method == "lime":
                start_time = time.time()
                local_exps = generate_lime_exp(inputs,features_names,class_names,mlp_model,task)
                end_time = time.time()
                time_dict[method] += (end_time-start_time)

            elif method == "sp_lime":
                start_time = time.time()
                global_exps = attr_to_dict(sp_lime(inputs,features_names,class_names,mlp_model,task))
                end_time = time.time()
                time_dict[method] += (end_time-start_time)

            elif method == "pfi":
                start_time = time.time()
                feature_permutation = FeaturePermutation(mlp_model)
                attribution = feature_permutation.attribute(inputs.float(),target = torch.argmax(mlp_model(inputs.float()),dim=1))
                attribution = torch.mean(torch.abs(attribution),dim=0)
                normalized_attribution = attribution/torch.sum(attribution)
                global_exps = attr_to_dict(create_feature_attribution_output(features_names,normalized_attribution))
                end_time = time.time()
                time_dict[method] += (end_time-start_time)

            elif method == "icc_topo":
                start_time = time.time()
                aug_scm = AugmentedSCM(scm_ = scm_model , mlp_ = mlp_model)
                causal_graph = config['causal_graph']
                icc_topo = torch.cat(Intrinsic_Causal_Contribution(neural_network=aug_scm,
                                topological_orederings= all_topological_sorts(causal_graph,n),
                                dim=n,rqmc= False, sample_size=50000
                                ),dim=0).to('cpu')

                global_exps = attr_to_dict(create_feature_attribution_output(features_names,icc_topo))
                # print(global_exps)
                end_time = time.time()
                time_dict[method] += (end_time-start_time)

            elif method == "icc_shap":
                start_time = time.time()
                aug_scm = AugmentedSCM(scm_ = scm_model , mlp_ = mlp_model)
                icc_shap = torch.cat(ICC_SHAP(dim=n,model=aug_scm,sample_size=50000,rqmc=False),dim=0).to('cpu')
                global_exps = attr_to_dict(create_feature_attribution_output(features_names,icc_shap))
                end_time = time.time()
                time_dict[method] += (end_time-start_time)

            else:
                # for "ig","itg","sg","shap"
                start_time = time.time()
                param_dict = openxai_config[method]
                param_dict = fill_param_dict(method, param_dict, inputs)
                explainer = Explainer(method=method, model=mlp_model, param_dict=param_dict)
                local_exps = explainer.get_explanations(inputs, labels).to('cpu')
                end_time = time.time()
                time_dict[method] += (end_time-start_time)

            if methods_type[method] == 'local':
                path = f'./explanations/local/{config['name']}/'
                if method == "lime":
                    local_df = pd.DataFrame(data = local_exps)
                else:
                    local_df = pd.DataFrame(data = local_exps.detach(),columns=features_names)
                path_to_csv = path + f'{config['name']}_{method}.csv'
                if not os.path.exists(path):
                    os.makedirs(path)
                local_df.to_csv(path_to_csv,index = False)
                #Generate GAM explanations
                start_time = time.time()
                gam = GAM(k=1, attributions_path=path_to_csv)
                gam.generate()
                global_exps = attr_to_dict(gam.explanations[0])
                end_time = time.time()
                time_dict[method] += (end_time-start_time)
                #Save GAM explanations
                path = f'./explanations/global/{config['name']}/'
                global_df = pd.DataFrame(data = global_exps)
                path_to_csv = path + f'{config['name']}_{method}.csv'
                if not os.path.exists(path):
                    os.makedirs(path)
                global_df.to_csv(path_to_csv,index = False)

            elif methods_type[method] == "global":
                path = f'./explanations/global/{config['name']}/'
                global_df = pd.DataFrame(data = global_exps)
                path_to_csv = path + f'{config['name']}_{method}.csv'
                if not os.path.exists(path):
                    os.makedirs(path)
                global_df.to_csv(path_to_csv,index = False)

            ge_dict[method] = global_exps
            ge = []
            for feature,attr in global_exps.items():
                global_explanations[feature].append(attr[0])
                ge.append(attr[0])
            ge = torch.tensor(ge).to('cpu')

            perturb_method =   Perturbation("tabular")
            evaluation_metrics[method]['pgu'] = dict()
            evaluation_metrics[method]['pgi'] = dict()
            for k in range(1,n+1):
                pgi = pred_faith(explanations= ge.repeat(inputs.shape[0],1).float().to('cpu'),
                            inputs=inputs.float(),task = task, targets= targets,model = mlp_model, k=k, perturb_method=perturb_method , feature_metadata=config['meta_data'], invert =False
                            )
                pgu = pred_faith(explanations= ge.repeat(inputs.shape[0],1).float().to('cpu'),
                            inputs=inputs.float(),task = task, targets= targets,model = mlp_model, k=k, perturb_method=perturb_method , feature_metadata=config['meta_data'], invert =True
                            )
                evaluation_metrics[method]['pgu'][f'k={k}'] = list((pgu[0].item(),pgu[1].item()))
                evaluation_metrics[method]['pgi'][f'k={k}'] = list((pgi[0].item(),pgi[1].item()))
    
    output_data = {
                    "ge_dict": ge_dict,
                    "evaluation_metrics": evaluation_metrics,
                    "time_dict": time_dict
                        }

    # Save to a JSON file
    fp = f"attributions/{config['name']}/"
    os.makedirs(fp,exist_ok=True)
    with open(fp + "attribution.json", "w") as f:
        json.dump(output_data, f, indent=4)

    return ge_dict, evaluation_metrics, time_dict, global_explanations

def generate_attr_plot(global_explanations,config):
    generate_causal_graph(config)
    methods = config['exp_methods']
    features_names = config['features_names']
    m = len(methods)
    n = len(features_names)
    ge_values = np.zeros((m, n))
    for i in range(n):
        ge_values[:, i] = global_explanations[features_names[i]]

    # Plotly grouped bar chart
    fig = go.Figure()

    color_blind_palette = [
    "#E69F00",  # Orange
    "#56B4E9",  # Light Blue
    "#009E73",  # Green
    "#F5C710",  # Amber
    "#0072B2",  # Blue
    "#999999",  # Grey
    "#000000",   # Black
    "#D55E00",  # Red
    "#CC79A7",  # Purple
]
    # Color scheme for the methods
    # colors = {
    # 'IG': 'rgb(21,138,106)',    # Light steel blue - professional, clear
    # 'IxG': 'rgb(23,108,155)',       # Light orange - warm, distinct
    # 'SG': 'rgb(194,135,32)',            # Sage green - fresh, readable
    # 'SHAP': 'rgb(186,97,26)',                   # Light olive - subtle yet visible
    # 'LIME': 'rgb(188,146,110)',                   # Light teal - modern, clean
    # 'SP LIME': 'rgb(193,130,181)',                # Light coral - distinct without being harsh
    # 'PFI': 'rgb(241,184,224)',  # Warm gray - neutral, balanced
    # 'ICC TOPO': 'rgb(148,148,148)',               # Coral orange
    # 'ICC SHAP': 'rgb(212,204,74)'                # Light brown - earthy, professional
    # }
    # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    # Update the trace colors in your code
    for i, method in enumerate(methods):
        fig.add_trace(go.Bar(
            x=features_names,
            y=ge_values[i],
            name=method_names[method],
            marker=dict(
                # color=colors[method_names[method]],
                color = color_blind_palette[i],
                line=dict(width=0.05)
            ),
            width=0.06
        ))

    # Update the layout to improve visualization
    fig.update_layout(
        # title='Global Explanations per Method for Auto Mpg dataset',
        # xaxis_title='Features',
        yaxis_title='Attributions',
        barmode='group',  # Group bars side-by-side
        bargap=0.45,  # Close the gap between bars
        bargroupgap=1,  # Add a gap between feature groups
        # xaxis_tickangle=-45,  # Rotate x-axis labels for readability
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(size=14),
        legend_title_text='Method',
        # Darken axes
         xaxis=dict(
        tickmode='array',
        tickvals=list(range(len(features_names))),  # Position ticks for each feature
        ticktext=features_names,  # Feature names as tick labels
        # tickangle=-45,  # Rotate tick labels for readability
        tickfont=dict(size=26, color='black', family='Times New Roman'),
        linecolor='lightgray',
        tickcolor='lightgray',
        range=[-0.5, len(features_names)-0.5],
        showgrid=False
        ),
        yaxis=dict(
        title=dict(font=dict(size=25, color='black', family='Times New Roman')),  # Dark, bold, and larger y-axis title
        linecolor='lightgray',
        tickcolor='lightgray',
        ticks='outside',
        tickfont=dict(size=25, color='black', family='Times New Roman'),  # Dark, bold y-axis tick labels
        showgrid=True,
        gridcolor='lightgray',
        dtick=0.1,
        range=[0, 0.9]
        ),
        legend=dict(
        font=dict(size=25, color='black', family='Times New Roman'),  # Dark, bold legend font
        title_font=dict(size=25, color='black', family='Times New Roman')  # Dark, bold legend title font
        ),
        showlegend=True
    )

    os.makedirs("assets",exist_ok=True)
    pio.write_image(fig, f'assets/{config['name']}_attributions.png', format='png', width=1000, height=800, scale=2)

    return fig



if __name__ == "__main__":
    from utils import CONFIG
    config = CONFIG['syn']
    mlp_path = '/home/dhruv/Files/Thesis/Dissertation/Code/models/syn_ann/syn_ann_seed_10/logs/version_0/checkpoints/epoch=161-step=5184.ckpt'
    scm_path = '/home/dhruv/Files/Thesis/Dissertation/Code/models/syn_nf/syn_nf_seed_10/logs/version_0/checkpoints/epoch=19-step=640.ckpt'
    mlp_model = MLPLightning.load_from_checkpoint(mlp_path,model = MLP(config['num_features'],config['hidden_layers_ann'],classification=False))
    scm_model = CausalNF.load_from_checkpoint(scm_path,flow = flow(config['num_features'],config['gd_adjacency'],config['hidden_layers_flow']))
    # print(mlp_model)
    # print(scm_model)
    print(generate_global_exps(config,mlp_model,scm_model))