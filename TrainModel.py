from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import CutClassifierModel
from Utils.Dataset import *
from Utils.util import *
import numpy as np
import os




# Dataset = TrackDataSet("MCDegredation_5")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/MCDegredation/LargeSamples/MC_Degredation5_TT_TrackNtuple.root",200000)#297676)
# Dataset.generate_test_train()
# Dataset.save_test_train_h5("MCDegredation_5/")

# Dataset = TrackDataSet("Degredation1_Train")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation1/Degredation1_Train_TrackNtuple.root",300000)#297676)
# Dataset.generate_test()
# Dataset.save_test_train_h5("Degredation1_Train/")

# Dataset = TrackDataSet("Degredation1_Zp")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation1/Degredation1_Zp_TrackNtuple.root",300000)#297676)
# Dataset.generate_test()
# Dataset.save_test_train_h5("Degredation1_Zp/")


folder_list = ["NoDegredation"]
name_list = ["No Degredation"]
[folder_list.append("Degredation"+str(i)) for i in range(1,10)] 
[name_list.append("Degredation "+str(i)) for i in range(1,10)] 

for i,folder in enumerate(folder_list):

    os.system("mkdir Projects/"+folder)
    os.system("mkdir Projects/"+folder+"/Plots")
    os.system("mkdir Projects/"+folder+"/FW")
    os.system("mkdir Projects/"+folder+"/Models")

    model = XGBoostClassifierModel()
    model.load_data("Datasets/"+folder+"/"+folder+"_Train/")
    model.train()
    model.save_model("Projects/"+folder+"/Models/",folder)
    model.load_model("Projects/"+folder+"/Models/",folder)
    model.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
    model.test()
    model.evaluate(plot=True,save_dir="Projects/"+folder+"/Plots/",name="Zp")
    plot_model(model,"Projects/"+folder+"/")
    plot_ROC_bins([calculate_ROC_bins(model.DataSet,model.y_predict_proba,variable="trk_eta",var_range=[-2.4,2.4],n_bins=20)],
                [name_list[i]],"Projects/"+folder+"/",variable="trk_eta",var_range=[-2.4,2.4],n_bins=20,typesetvar="Track $\\eta$")
    plot_ROC_bins([calculate_ROC_bins(model.DataSet,model.y_predict_proba,variable="trk_pt",var_range=[2,100],n_bins=10)],
                [name_list[i]],"Projects/"+folder+"/",variable="trk_pt",var_range=[2,100],n_bins=10,typesetvar="Track $p_T$")
    plot_ROC_bins([calculate_ROC_bins(model.DataSet,model.y_predict_proba,variable="trk_phi",var_range=[-3.14,3.14],n_bins=20)],
                [name_list[i]],"Projects/"+folder+"/",variable="trk_phi",var_range=[-3.14,3.14],n_bins=20,typesetvar="Track $\\phi$")
    plot_ROC_bins([calculate_ROC_bins(model.DataSet,model.y_predict_proba,variable="trk_z0",var_range=[-15,15],n_bins=20)],
                [name_list[i]],"Projects/"+folder+"/",variable="trk_z0",var_range=[-15,15],n_bins=20,typesetvar="Track $z_0$")
    plot_ROC_bins([calculate_ROC_bins(model.DataSet,model.y_predict_proba,variable="trk_MVA1",var_range=[0,1],n_bins=20)],
                [name_list[i]],"Projects/"+folder+"/",variable="trk_MVA1",var_range=[0,1],n_bins=20,typesetvar="Track MVA")
    precisions = ['ap_fixed<12,6>','ap_fixed<11,6>','ap_fixed<11,5>','ap_fixed<10,6>','ap_fixed<10,5>','ap_fixed<10,4>']
    synth_model(model,sim=True,hdl=True,hls=True,cpp=False,onnx=False,
                test_events=10000,
                precisions=precisions,
                save_dir="Projects/"+folder+"/")
