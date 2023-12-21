from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import Binned_CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

setmatplotlib()

# This evaluates one model on all the datasets in folder_list

#folder_list = ["Degradation0","Degradation1","Degradation5","Degradation10"]
folder_list = ["Degradation10"]
#model_name = "Retrained_new"
#name = "Model retrained new"

model_name = "Degradation0"
name = "Model Trained on No Degradation"

plot_types = ["ROC","FPR","TPR","score"]

for i,folder in enumerate(folder_list):

    model = XGBoostClassifierModel(model_name)
    model.load_model("Projects/"+model_name+"/Models/",model_name+"_XGB")
    model.load_data("/eos/user/k/klaw/ML_L1_workshop/TrackQuality_package/"+folder+"/"+folder+"_Test/")
    model.test()
    model.evaluate(plot=True,save_dir="Projects/"+model_name+"/Plots/"+folder+"/",full_parameter_rocs=True)

    for plottype in plot_types:
        threshold = 0.7
        plot_ROC_bins([model.eta_roc_dict],
                      [name],
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["eta"]["branch"],
                    var_range=Parameter_config["eta"]["range"],
                    n_bins=Parameter_config["eta"]["bins"],
                    typesetvar=Parameter_config["eta"]["typeset"],
                    what=plottype, threshold = threshold)
                    
        plot_ROC_bins([model.eta_pt_dict],
                      [name],
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["pt"]["branch"],
                    var_range=Parameter_config["pt"]["range"],
                    n_bins=Parameter_config["pt"]["bins"],
                    typesetvar=Parameter_config["pt"]["typeset"],
                    what=plottype, threshold = threshold)
        
        plot_ROC_bins([model.eta_phi_dict],
                      [name],
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["phi"]["branch"],
                    var_range=Parameter_config["phi"]["range"],
                    n_bins=Parameter_config["phi"]["bins"],
                    typesetvar=Parameter_config["phi"]["typeset"],
                    what=plottype, threshold = threshold)
        
        plot_ROC_bins([model.eta_z0_dict],
                     [name],
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["z0"]["branch"],
                    var_range=Parameter_config["z0"]["range"],
                    n_bins=Parameter_config["z0"]["bins"],
                    typesetvar=Parameter_config["z0"]["typeset"],
                    what=plottype, threshold = threshold)
    
    plot_ROC([model.roc_dict],[name],"Projects/"+model_name+"/Plots/"+folder+"/")
    
