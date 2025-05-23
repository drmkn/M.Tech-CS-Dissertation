import numpy as np
import pandas as pd
import torch,os
from helper import generate_lime_exp,torch_to_csv,sp_lime,attr_to_dict,create_feature_attribution_output
from helper import AugmentedSCM
from flow import causal_graphs,adjacency,num_features,meta_datas
from OpenXAI.openxai import Explainer
from OpenXAI.openxai.experiment_utils import load_config,fill_param_dict
from global_attribution_mapping.gam.gam import GAM
from captum.attr import FeaturePermutation
from utils import Intrinsic_Causal_Contribution, all_topological_sorts,ICC_SHAP
from helper import create_feature_attribution_output
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
from OpenXAI.openxai.metrics import eval_pred_faithfulness, eval_relative_stability
from OpenXAI.openxai.evaluator import Evaluator
from helper import Perturbation, pred_faith
from OpenXAI.openxai.explainers.perturbation_methods import get_perturb_method
from OpenXAI.openxai.experiment_utils import generate_mask
import time
STD= 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_indices = {0:"ig", 1:"itg", 2:"sg", 3:"shap", 4:"lime", 5:"sp_lime", 6:"pfi", 7:"icc_topo", 8:"icc_shap"}
methods_type = {"ig":'local',"itg":'local',"sg":'local',"shap":'local',"lime":'local',"sp_lime":'global',"pfi":'global',"icc_topo":'global',"icc_shap":'global'}
methods = list(exp_indices.values())
method_names = {"ig":'IG',"itg":'IxG',"sg":'SG',"shap":'SHAP',"lime":'LIME',"sp_lime":'SP LIME',"pfi":'PFI',"icc_topo":'ICC TOPO',"icc_shap":'ICC SHAP'}
# methods = ["icc_topo","icc_shap"]
# print(methods)
# methods.remove("sp_lime")
def generate_global_exps(config, ann_model, scm_model, data_name, inputs, labels,targets, features_names):

    if data_name in ['syn','auto_mpg']:
        task = 'regression'
        class_names = None
    else:
        task = 'classification'
        class_names = ['0','1']  

    m = len(methods)
    n = len(features_names)
    global_explanations = dict()
    evaluation_metrics = dict()
    Time_dict = dict()
    icc_dict = dict()

    # Initialize the global_explanations dictionary with features as keys
    for feature in features_names:
        global_explanations[feature] = []
    for method in methods:
        evaluation_metrics[method] = dict() 
        Time_dict[method] = 0
        icc_dict[method] = dict()




    rqmcs = [False]
    for method in methods:
        for rqmc in rqmcs:
            if method == "lime":
                start_time = time.time()
                local_exps = generate_lime_exp(inputs,features_names,class_names,ann_model,task)
                end_time = time.time()
                Time_dict[method] += (end_time-start_time)

            elif method == "sp_lime":
                # print(sp_lime(inputs,features_names,class_names,ann_model,task))
                # print(class_names)
                start_time = time.time()
                global_exps = attr_to_dict(sp_lime(inputs,features_names,class_names,ann_model,task))
                end_time = time.time()
                Time_dict[method] += (end_time-start_time)

            elif method == "pfi":
                start_time = time.time()
                feature_permutation = FeaturePermutation(ann_model)
                attribution = feature_permutation.attribute(inputs.float(),target = torch.argmax(ann_model(inputs.float()),dim=1))
                attribution = torch.mean(torch.abs(attribution),dim=0)
                normalized_attribution = attribution/torch.sum(attribution)
                global_exps = attr_to_dict(create_feature_attribution_output(features_names,normalized_attribution))
                end_time = time.time()
                Time_dict[method] += (end_time-start_time)

            elif method == "icc_topo":
                start_time = time.time()
                aug_scm = AugmentedSCM(scm_ = scm_model , classifier_ = ann_model)
                causal_graph = causal_graphs[data_name]
                icc_topo = torch.cat(Intrinsic_Causal_Contribution(neural_network=aug_scm,
                                topological_orederings= all_topological_sorts(causal_graph,n),
                                dim=n,rqmc= rqmc, sample_size=50000
                                ),dim=0).to('cpu')

                global_exps = attr_to_dict(create_feature_attribution_output(features_names,icc_topo))
                # print(global_exps)
                end_time = time.time()
                Time_dict[method] += (end_time-start_time)

            elif method == "icc_shap":
                start_time = time.time()
                aug_scm = AugmentedSCM(scm_ = scm_model , classifier_ = ann_model)
                icc_shap = torch.cat(ICC_SHAP(dim=n,model=aug_scm,sample_size=50000,rqmc=rqmc),dim=0).to('cpu')
                global_exps = attr_to_dict(create_feature_attribution_output(features_names,icc_shap))
                end_time = time.time()
                Time_dict[method] += (end_time-start_time)

            else:
                # for "ig","itg","sg","shap"
                start_time = time.time()
                param_dict = load_config(config)['explainers'][method]
                param_dict = fill_param_dict(method, param_dict, inputs)
                explainer = Explainer(method=method, model=ann_model, param_dict=param_dict)
                local_exps = explainer.get_explanations(inputs, labels).to('cpu')
                end_time = time.time()
                Time_dict[method] += (end_time-start_time)

            if methods_type[method] == 'local':
                path = f'./explanations/local/{data_name}/'
                if method == "lime":
                    local_df = pd.DataFrame(data = local_exps)
                else:
                    local_df = pd.DataFrame(data = local_exps.detach(),columns=features_names)
                path_to_csv = path + f'{data_name}_{method}.csv'
                if not os.path.exists(path):
                    os.makedirs(path)
                local_df.to_csv(path_to_csv,index = False)
                #Generate GAM explanations
                start_time = time.time()
                gam = GAM(k=1, attributions_path=path_to_csv)
                gam.generate()
                global_exps = attr_to_dict(gam.explanations[0])
                end_time = time.time()
                Time_dict[method] += (end_time-start_time)
                #Save GAM explanations
                path = f'./explanations/global/{data_name}/'
                global_df = pd.DataFrame(data = global_exps)
                path_to_csv = path + f'{data_name}_{method}.csv'
                if not os.path.exists(path):
                    os.makedirs(path)
                global_df.to_csv(path_to_csv,index = False)

            elif methods_type[method] == "global":
                path = f'./explanations/global/{data_name}/'
                global_df = pd.DataFrame(data = global_exps)
                path_to_csv = path + f'{data_name}_{method}.csv'
                if not os.path.exists(path):
                    os.makedirs(path)
                global_df.to_csv(path_to_csv,index = False)

            # print(global_exps)
            icc_dict[method][rqmc] = global_exps
            ge = []
            for feature,attr in global_exps.items():
                global_explanations[feature].append(attr[0])
                ge.append(attr[0])
            ge = torch.tensor(ge).to('cpu')

            perturb_method =   Perturbation("tabular")
            # print(ge.repeat(inputs.shape[0],1).shape,inputs.shape)
            for k in range(1,n+1):
                # pgi = pred_faith(explanations= ge.repeat(inputs.shape[0],1).float().to('cpu'),
                #             inputs=inputs.float(),task = task, targets= targets,model = ann_model, k=k, perturb_method=perturb_method , feature_metadata=meta_datas[data_name], invert =False
                #             )
                pgu = pred_faith(explanations= ge.repeat(inputs.shape[0],1).float().to('cpu'),
                            inputs=inputs.float(),task = task, targets= targets,model = ann_model, k=k, perturb_method=perturb_method , feature_metadata=meta_datas[data_name], invert =True
                            )
                evaluation_metrics[method][f'k={k}'] = list((pgu[0].item(),pgu[1].item()))
                # evaluation_metrics[method][f'k={k}'].append(pgi.item())
                # evaluation_metrics[method][f'k={k}'].append(pgu.item())
  
    return global_explanations,evaluation_metrics,Time_dict,icc_dict

# def generate_attr_plot(global_explanations,features_names):
#     # Convert global_explanations dictionary to a 2D array for plotting
#     # methods = ['icc_topo_false','icc_topo_true','icc_shap_false','icc_shap_true']
#     # method_names = {'icc_topo_false': 'ICC_TOPO(rqmc = False)','icc_topo_true':'ICC_TOPO(rqmc = True)','icc_shap_false':'ICC_SHAP(rqmc = False)','icc_shap_true':'ICC_SHAP(rqmc = True)'}
#     m = len(methods)
#     n = len(features_names)
#     ge_values = np.zeros((m, n))
#     for i in range(n):
#         ge_values[:, i] = global_explanations[features_names[i]]

#     # Plotly grouped bar chart
#     fig = go.Figure()
#     # method_names = {"ig":'Integrated Gradient',"itg":'Gradient * Input',"sg":'Smooth Grad',"shap":'SHAP',"lime":'LIME',"sp_lime":'SP LIME',"pfi":'Permutation Feature Importance',"icc_topo":'ICC_TOPO',"icc_shap":'ICC_SHAP'}


#     color_blind_palette = [
#     "#E69F00",  # Orange
#     "#56B4E9",  # Light Blue
#     "#009E73",  # Green
#     "#F5C710",  # Amber
#     "#0072B2",  # Blue
#     "#999999",  # Grey
#     "#000000",   # Black
#     "#D55E00",  # Red
#     "#CC79A7",  # Purple
# ]
#     # Color scheme for the methods
#     # colors = {
#     # 'IG': 'rgb(21,138,106)',    # Light steel blue - professional, clear
#     # 'IxG': 'rgb(23,108,155)',       # Light orange - warm, distinct
#     # 'SG': 'rgb(194,135,32)',            # Sage green - fresh, readable
#     # 'SHAP': 'rgb(186,97,26)',                   # Light olive - subtle yet visible
#     # 'LIME': 'rgb(188,146,110)',                   # Light teal - modern, clean
#     # 'SP LIME': 'rgb(193,130,181)',                # Light coral - distinct without being harsh
#     # 'PFI': 'rgb(241,184,224)',  # Warm gray - neutral, balanced
#     # 'ICC TOPO': 'rgb(148,148,148)',               # Coral orange
#     # 'ICC SHAP': 'rgb(212,204,74)'                # Light brown - earthy, professional
#     # }
#     # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

#     # Update the trace colors in your code
#     for i, method in enumerate(methods):
#         fig.add_trace(go.Bar(
#             x=features_names,
#             y=ge_values[i],
#             name=method_names[method],
#             marker=dict(
#                 # color=colors[method_names[method]],
#                 color = color_blind_palette[i],
#                 line=dict(width=0.05)
#             ),
#             width=0.06
#         ))

#     # Update the layout to improve visualization
#     fig.update_layout(
#         # title='Global Explanations per Method for Auto Mpg dataset',
#         # xaxis_title='Features',
#         yaxis_title='Attributions',
#         barmode='group',  # Group bars side-by-side
#         bargap=0.45,  # Close the gap between bars
#         bargroupgap=1,  # Add a gap between feature groups
#         # xaxis_tickangle=-45,  # Rotate x-axis labels for readability
#         plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
#         font=dict(size=14),
#         legend_title_text='Method',
#         # Darken axes
#          xaxis=dict(
#         tickmode='array',
#         tickvals=list(range(len(features_names))),  # Position ticks for each feature
#         ticktext=features_names,  # Feature names as tick labels
#         # tickangle=-45,  # Rotate tick labels for readability
#         tickfont=dict(size=26, color='black', family='Times New Roman'),
#         linecolor='lightgray',
#         tickcolor='lightgray',
#         range=[-0.5, len(features_names)-0.5],
#         showgrid=False
#         ),
#         yaxis=dict(
#         title=dict(font=dict(size=25, color='black', family='Times New Roman')),  # Dark, bold, and larger y-axis title
#         linecolor='lightgray',
#         tickcolor='lightgray',
#         ticks='outside',
#         tickfont=dict(size=25, color='black', family='Times New Roman'),  # Dark, bold y-axis tick labels
#         showgrid=True,
#         gridcolor='lightgray',
#         dtick=0.1,
#         range=[0, 0.9]
#         ),
#         legend=dict(
#         font=dict(size=25, color='black', family='Times New Roman'),  # Dark, bold legend font
#         title_font=dict(size=25, color='black', family='Times New Roman')  # Dark, bold legend title font
#         ),
#         showlegend=False
#     )

#     # Show the plot
#     fig.show()

#     return fig

def generate_attr_plot(global_explanations, features_names):
    # Convert global_explanations dictionary to a 2D array for plotting
    m = len(methods)
    n = len(features_names)
    ge_values = np.zeros((m, n))
    for i in range(n):
        ge_values[:, i] = global_explanations[features_names[i]]

    # Plotly grouped bar chart
    fig = go.Figure()

    # Color-blind-friendly palette
    color_blind_palette = [
        "#E69F00",  # Orange
        "#56B4E9",  # Light Blue
        "#009E73",  # Green
        "#F5C710",  # Amber
        "#0072B2",  # Blue
        "#999999",  # Grey
        "#000000",  # Black
        "#D55E00",  # Red
        "#CC79A7",  # Purple
    ]

    # Add bars for each method
    for i, method in enumerate(methods):
        fig.add_trace(go.Bar(
            x=features_names,
            y=ge_values[i],
            name=method_names[method],
            marker=dict(
                color=color_blind_palette[i],
                line=dict(width=0.05)
            ),
            width=0.1  # Increase bar width to reduce spacing between bars
        ))

    # Update the layout to improve visualization
    fig.update_layout(
        # yaxis_title='Attributions',
        barmode='group',  # Group bars side-by-side
        bargap=0.2,  # Reduce spacing between bars (default is 0.2)
        bargroupgap=0.1,  # Reduce spacing between groups of bars
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(size=14),
        legend_title_text='Method',

        # X-axis settings
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(features_names))),  # Align ticks to bars
            ticktext=features_names,  # Feature names as tick labels
            tickfont=dict(size=30, color='black', family='Times New Roman'),
            linecolor='lightgray',
            tickcolor='lightgray',
            range=[-0.5, len(features_names)-0.5],  # Adjust axis range
            showgrid=False
        ),
        # Y-axis settings
        yaxis=dict(
            title=dict(font=dict(size=25, color='black', family='Times New Roman')),
            linecolor='lightgray',
            tickcolor='lightgray',
            ticks='outside',
            tickfont=dict(size=29, color='black', family='Times New Roman'),
            showgrid=True,
            gridcolor='lightgray',
            dtick=0.1,
            range=[0, 0.9]
        ),
        # Legend settings
        legend=dict(
            font=dict(size=29, color='black', family='Times New Roman'),
            title_font=dict(size=29, color='black', family='Times New Roman')
        ),
        showlegend=False
    )

    # Show the plot
    fig.show()

    return fig

