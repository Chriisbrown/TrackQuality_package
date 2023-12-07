from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import Binned_CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

setmatplotlib()

folder_list = ["NoDegredation"]
name_list = ["NoDegredation"]
# [folder_list.append("Degredation"+str(i)) for i in range(1,10)] 
# [name_list.append("Degredation "+str(i)) for i in range(1,10)] 

plot_types = ["ROC","FPR","TPR","score"]

for i,folder in enumerate(folder_list):
    os.system("mkdir Projects/"+folder)
    os.system("mkdir Projects/"+folder+"/Plots")
    os.system("mkdir Projects/"+folder+"/FW")
    os.system("mkdir Projects/"+folder+"/Models")

    # cutmodel = Binned_CutClassifierModel("Cut")
    # cutmodel.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
    # cutmodel.test()
    # cutmodel.evaluate(plot=False,binned=False)
    # cutmodel.full_save("Projects/"+folder+"/Models/"+folder+"_Cut/","Cut_")
    # cutmodel.full_load("Projects/"+folder+"/Models/"+folder+"_Cut/","Cut_")

    model = XGBoostClassifierModel(name_list[i])
    model.load_data("Datasets/"+folder+"/"+folder+"_Train/")
    #model.train()
    #model.save_model("Projects/"+folder+"/Models/",folder+"_XGB_")
    model.load_model("Projects/"+folder+"/Models/",folder+"_XGB_")

    model.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
    model.test()
    #model.evaluate(plot=True,save_dir="Projects/"+folder+"/Plots/")
    #model.full_save("Projects/"+folder+"/Models/"+folder+"/",folder+"_XGB_")
    #model.full_load("Projects/"+folder+"/Models/"+folder+"/",folder+"_XGB_")

    #plot_model(model,"Projects/"+folder+"/")
    '''
    for plottype in plot_types:
        threshold = 0.55
        plot_ROC_bins([model.eta_roc_dict,cutmodel.eta_roc_dict],
                    [name_list[i]+" XGB threshold = "+str(threshold),name_list[i]+" Cut"],
                    "Projects/"+folder+"/",
                    variable=Parameter_config["eta"]["branch"],
                    var_range=Parameter_config["eta"]["range"],
                    n_bins=Parameter_config["eta"]["bins"],
                    typesetvar=Parameter_config["eta"]["typeset"],
                    what=plottype, threshold = threshold)
                    
        plot_ROC_bins([model.pt_roc_dict,cutmodel.pt_roc_dict],
                    [name_list[i]+" XGB threshold = "+str(threshold),name_list[i]+" Cut"],
                    "Projects/"+folder+"/",
                    variable=Parameter_config["pt"]["branch"],
                    var_range=Parameter_config["pt"]["range"],
                    n_bins=Parameter_config["pt"]["bins"],
                    typesetvar=Parameter_config["pt"]["typeset"],
                    what=plottype, threshold = threshold)
        
        plot_ROC_bins([model.phi_roc_dict,cutmodel.phi_roc_dict],
                    [name_list[i]+" XGB threshold = "+str(threshold),name_list[i]+" Cut"],
                    "Projects/"+folder+"/",
                    variable=Parameter_config["phi"]["branch"],
                    var_range=Parameter_config["phi"]["range"],
                    n_bins=Parameter_config["phi"]["bins"],
                    typesetvar=Parameter_config["phi"]["typeset"],
                    what=plottype, threshold = threshold)
        
        plot_ROC_bins([model.z0_roc_dict,cutmodel.z0_roc_dict],
                    [name_list[i]+" XGB threshold = "+str(threshold),name_list[i]+" Cut"],
                    "Projects/"+folder+"/",
                    variable=Parameter_config["z0"]["branch"],
                    var_range=Parameter_config["z0"]["range"],
                    n_bins=Parameter_config["z0"]["bins"],
                    typesetvar=Parameter_config["z0"]["typeset"],
                    what=plottype, threshold = threshold)
    
    plot_ROC([model.roc_dict,cutmodel.roc_dict],[name_list[i]+" XGB",name_list[i]+" Cut"],"Projects/"+folder+"/")
    '''
    precisions = ['ap_fixed<12,6>','ap_fixed<11,6>','ap_fixed<11,5>','ap_fixed<10,6>','ap_fixed<10,5>','ap_fixed<10,4>']
    #precisions = ['ap_fixed<12,6>','ap_fixed<10,5>']

    synth_model(model,sim=True,hdl=True,hls=True,cpp=True,onnx=True,python=True,
                 test_events=10000,
                 precisions=precisions,
                 save_dir="Projects/"+folder+"/")
