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


# class Stability_Perturbation(BasePerturbation):
#     def __init__(self, data_format,std =0.05):
#         self.std = std
#         super(Stability_Perturbation, self).__init__(data_format)

#     def get_perturbed_inputs(self, original_sample: torch.FloatTensor, feature_mask: torch.BoolTensor,
#                              num_samples: int, feature_metadata: list) -> torch.tensor:
#         '''
#         feature mask : this indicates the static features
#         num_samples : number of perturbed samples.
#         '''
#         feature_type = feature_metadata
#         assert len(feature_mask) == len(original_sample),\
#             f"mask size == original sample in get_perturbed_inputs for {self.__class__}"
        
        
#         # Processing continuous columns
#         perturbations =   torch.normal(0,self.std,[num_samples, len(feature_type)])
        
#         # keeping features static that are in top-K based on feature mask
#         perturbed_samples = original_sample * feature_mask  + perturbations * (~feature_mask)
        
#         return torch.clamp(perturbed_samples, min=0.0, max=1.0)
    



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






