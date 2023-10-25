from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import CutClassifierModel
from Utils.Dataset import *
from Utils.util import *
import numpy as np
import os




Dataset = TrackDataSet("MCDegredation_5")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/MCDegredation/LargeSamples/MC_Degredation5_TT_TrackNtuple.root",200000)#297676)
Dataset.generate_test_train()
Dataset.save_test_train_h5("MCDegredation_5/")

# Dataset = TrackDataSet("NoDegredation_Zp")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/NoDegredation/NoDegredation_Zp_TrackNtuple.root",300000)#297676)
# Dataset.generate_test()
# Dataset.save_test_train_h5("NoDegredation_Zp/")
'''
NoDegredation = XGBoostClassifierModel()
#NoDegredation.load_data("NoDegredation_Train/")
#NoDegredation.train()#
#NoDegredation.save_model("NoDegredation/Models/","NoDegredation")
NoDegredation.load_model("Projects/NoDegredation/Models/","NoDegredation")
NoDegredation.load_data("Datasets/NoDegredation_Zp/")
NoDegredation.test()
NoDegredation.evaluate(plot=True,save_dir="Projects/NoDegredation/",name="Zp")
plot_model(NoDegredation,"Projects/NoDegredation/")
# plot_ROC_bins([calculate_ROC_bins(NoDegredation.DataSet,NoDegredation.y_predict_proba,variable="trk_eta",var_range=[-2.4,2.4],n_bins=20)],
#               ["No Degradation"],"NoDegredation/",variable="trk_eta",var_range=[-2.4,2.4],n_bins=20,typesetvar="Track $\\eta$")
plot_ROC_bins([calculate_ROC_bins(NoDegredation.DataSet,NoDegredation.y_predict_proba,variable="trk_pt",var_range=[2,100],n_bins=10)],
              ["No Degradation"],"Projects/NoDegredation/",variable="trk_pt",var_range=[2,100],n_bins=10,typesetvar="Track $p_T$")
plot_ROC_bins([calculate_ROC_bins(NoDegredation.DataSet,NoDegredation.y_predict_proba,variable="trk_phi",var_range=[-3.14,3.14],n_bins=20)],
              ["No Degradation"],"Projects/NoDegredation/",variable="trk_phi",var_range=[-3.14,3.14],n_bins=20,typesetvar="Track $\\phi$")
plot_ROC_bins([calculate_ROC_bins(NoDegredation.DataSet,NoDegredation.y_predict_proba,variable="trk_z0",var_range=[-15,15],n_bins=20)],
              ["No Degradation"],"Projects/NoDegredation/",variable="trk_z0",var_range=[-15,15],n_bins=20,typesetvar="Track $z_0$")
plot_ROC_bins([calculate_ROC_bins(NoDegredation.DataSet,NoDegredation.y_predict_proba,variable="trk_MVA1",var_range=[0,1],n_bins=20)],
              ["No Degradation"],"Projects/NoDegredation/",variable="trk_MVA1",var_range=[0,1],n_bins=20,typesetvar="Track MVA")
#precisions = ['ap_fixed<12,6>','ap_fixed<11,6>','ap_fixed<11,5>','ap_fixed<10,6>','ap_fixed<10,5>','ap_fixed<10,4>']
# synth_model(NoDegredation,sim=True,hdl=True,hls=True,cpp=False,onnx=False,
#             test_events=10000,
#             precisions=['ap_fixed<12,6>','ap_fixed<10,4>'],
#             save_dir="NoDegredation/")
'''
