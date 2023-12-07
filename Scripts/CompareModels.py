from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import Binned_CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

setmatplotlib()

folder_list = ["NoDegredation","Degredation3","Degredation9"]
name_list = ["NoDegredation","Degredation3","Degredation9"]
# [folder_list.append("Degredation"+str(i)) for i in range(1,10)] 
# [name_list.append("Degredation "+str(i)) for i in range(1,10)] 

plot_types = ["ROC","FPR","TPR","score"]

for i,folder in enumerate(folder_list):
    NDmodel = XGBoostClassifierModel(name_list[i])
    NDmodel.load_model("Projects/NoDegredation/Models/","NoDegredation_XGB_")
    NDmodel.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
    NDmodel.test()
    NDmodel.evaluate(plot=False,save_dir="Projects/"+folder+"/Plots/")

    model3 = XGBoostClassifierModel("Degredation3")
    #model2.train()
    #model2.save_model("Projects/"+folder+"/Models/",folder+"_XGB_")
    model3.load_model("Projects/Degredation3/Models/","Degredation3_XGB_")

    model3.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
    model3.test()
    model3.evaluate(plot=False,save_dir="Projects/"+folder+"/Plots/")

    model9 = XGBoostClassifierModel("Degredation9")
    #model2.train()
    #model2.save_model("Projects/"+folder+"/Models/",folder+"_XGB_")
    model9.load_model("Projects/Degredation9/Models/","Degredation9")

    model9.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
    model9.test()
    model9.evaluate(plot=False,save_dir="Projects/"+folder+"/Plots/")


    for plottype in plot_types:
        threshold = 0.7
        plot_ROC_bins([NDmodel.eta_roc_dict,model3.eta_roc_dict,model9.eta_roc_dict],
                    ["No Degredation BDT","Degredation 3 BDT","Degredation 9 BDT"],
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["eta"]["branch"],
                    var_range=Parameter_config["eta"]["range"],
                    n_bins=Parameter_config["eta"]["bins"],
                    typesetvar=Parameter_config["eta"]["typeset"],
                    what=plottype, threshold = threshold)
                    
        plot_ROC_bins([NDmodel.pt_roc_dict,model3.pt_roc_dict,model9.pt_roc_dict],
                    ["No Degredation BDT","Degredation 3 BDT","Degredation 9 BDT"],
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["pt"]["branch"],
                    var_range=Parameter_config["pt"]["range"],
                    n_bins=Parameter_config["pt"]["bins"],
                    typesetvar=Parameter_config["pt"]["typeset"],
                    what=plottype, threshold = threshold)
        
        plot_ROC_bins([NDmodel.phi_roc_dict,model3.phi_roc_dict,model9.phi_roc_dict],
                    ["No Degredation BDT","Degredation 3 BDT","Degredation 9 BDT"],
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["phi"]["branch"],
                    var_range=Parameter_config["phi"]["range"],
                    n_bins=Parameter_config["phi"]["bins"],
                    typesetvar=Parameter_config["phi"]["typeset"],
                    what=plottype, threshold = threshold)
        
        plot_ROC_bins([NDmodel.z0_roc_dict,model3.z0_roc_dict,model9.z0_roc_dict],
                    ["No Degredation BDT","Degredation 3 BDT","Degredation 9 BDT"],
                    "Projects/Scan/"+folder+"/",
                    variable=Parameter_config["z0"]["branch"],
                    var_range=Parameter_config["z0"]["range"],
                    n_bins=Parameter_config["z0"]["bins"],
                    typesetvar=Parameter_config["z0"]["typeset"],
                    what=plottype, threshold = threshold)
    
    plot_ROC([NDmodel.roc_dict,model3.roc_dict,model9.roc_dict],["No Degredation BDT","Degredation 3 BDT","Degredation 9 BDT"],"Projects/Scan/"+folder+"/")
    
    # precisions = ['ap_fixed<12,6>','ap_fixed<11,6>','ap_fixed<11,5>','ap_fixed<10,6>','ap_fixed<10,5>','ap_fixed<10,4>']
    # synth_model(model,sim=True,hdl=True,hls=True,cpp=False,onnx=False,
    #             test_events=10000,
    #             precisions=precisions,
    #             save_dir="Projects/"+folder+"/")
