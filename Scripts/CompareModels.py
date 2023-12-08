from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import Binned_CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

setmatplotlib()

folder_list = ["Degradation0","Degradation1","Degradation5","Degradation10"]
model_list = ["Degradation0","Degradation1","Degradation5","Degradation10"]
name_list = ["No Degradation","1% Bad Stubs","5% Bad Stubs","10% Bad Stubs"]

folder_list = ["Degradation0","Degradation1"]
model_list = ["Degradation0","Degradation1"]
name_list = ["No Degradation","1% Bad Stubs"]

plot_types = ["ROC","FPR","TPR","score"]

for i,folder in enumerate(folder_list):
    models = []
    for model_name in model_list:

        model = XGBoostClassifierModel(model_name)
        model.load_model("Projects/"+model_name+"/Models/",model_name+"_XGB")
        model.load_data("Datasets/"+folder+"/"+folder+"_Test/")
        model.test()
        model.evaluate(plot=False,save_dir="Projects/"+folder+"/Plots/",full_parameter_rocs=True)

        models.append(model)

    for plottype in plot_types:
        threshold = 0.7
        plot_ROC_bins([model.eta_roc_dict for model in models],
                    name_list,
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["eta"]["branch"],
                    var_range=Parameter_config["eta"]["range"],
                    n_bins=Parameter_config["eta"]["bins"],
                    typesetvar=Parameter_config["eta"]["typeset"],
                    what=plottype, threshold = threshold)
                    
        plot_ROC_bins([model.eta_roc_dict for model in models],
                    name_list,
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["pt"]["branch"],
                    var_range=Parameter_config["pt"]["range"],
                    n_bins=Parameter_config["pt"]["bins"],
                    typesetvar=Parameter_config["pt"]["typeset"],
                    what=plottype, threshold = threshold)
        
        plot_ROC_bins([model.eta_roc_dict for model in models],
                    name_list,
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["phi"]["branch"],
                    var_range=Parameter_config["phi"]["range"],
                    n_bins=Parameter_config["phi"]["bins"],
                    typesetvar=Parameter_config["phi"]["typeset"],
                    what=plottype, threshold = threshold)
        
        plot_ROC_bins([model.eta_roc_dict for model in models],
                    name_list,
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["z0"]["branch"],
                    var_range=Parameter_config["z0"]["range"],
                    n_bins=Parameter_config["z0"]["bins"],
                    typesetvar=Parameter_config["z0"]["typeset"],
                    what=plottype, threshold = threshold)
    
    plot_ROC([model.eta_roc_dict for model in models],name_list,"Projects/Scan/"+folder+"/")
    
