from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import Binned_CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

setmatplotlib()

folder_list = ["Degradation0","Degradation1","Degradation5","Degradation10"]
name_list = ["Degradation0","Degradation1","Degradation5","Degradation10"]
# [folder_list.append("Degradation"+str(i)) for i in range(1,10)] 
# [name_list.append("Degradation "+str(i)) for i in range(1,10)] 

plot_types = ["ROC","FPR","TPR","score"]

for i,folder in enumerate(folder_list):
    os.system("mkdir Projects/"+folder)
    os.system("mkdir Projects/"+folder+"/Plots")
    os.system("mkdir Projects/"+folder+"/FW")
    os.system("mkdir Projects/"+folder+"/Models")

    # cutmodel = Binned_CutClassifierModel("Cut")
    # cutmodel.load_data("Datasets/"+folder+"/"+folder+"_Test/")
    # cutmodel.test()
    # cutmodel.evaluate(plot=False,binned=False)
    # cutmodel.full_save("Projects/"+folder+"/Models/"+folder+"_Cut/","Cut")
    # cutmodel.full_load("Projects/"+folder+"/Models/"+folder+"_Cut/","Cut")

    model = XGBoostClassifierModel(name_list[i])
    model.load_data("Datasets/"+folder+"/"+folder+"_Train/")
    model.train()
    model.save_model("Projects/"+folder+"/Models/",folder+"_XGB")
    model.load_model("Projects/"+folder+"/Models/",folder+"_XGB")

    model.load_data("Datasets/"+folder+"/"+folder+"_Test/")
    model.test()
    model.evaluate(plot=True,save_dir="Projects/"+folder+"/Plots/")
    model.full_save("Projects/"+folder+"/Models/"+folder+"/",folder+"_XGB")
    model.full_load("Projects/"+folder+"/Models/"+folder+"/",folder+"_XGB")

    plot_model(model,"Projects/"+folder+"/")
 
    # precisions = ['ap_fixed<12,6>','ap_fixed<11,6>','ap_fixed<11,5>','ap_fixed<10,6>','ap_fixed<10,5>','ap_fixed<10,4>']
    precisions = ['ap_fixed<12,6>','ap_fixed<10,5>']

    synth_model(model,sim=True,hdl=True,hls=True,cpp=True,onnx=True,python=True,
                    test_events=10000,
                    precisions=precisions,
                    save_dir="Projects/"+folder+"/")
