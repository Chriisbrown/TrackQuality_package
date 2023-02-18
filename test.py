from GBDTTrackQualityModel import XGBoostClassifierModel,SklearnClassifierModel
from CutTrackQualityModel import CutClassifierModel
from Dataset import *
import numpy as np
import os

# AllKFDataset = KFDataSet("KFDataset_WithAll_9K",withstubs=True,withchi=True,withmatrix=True)
# AllKFDataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
# AllKFDataset.generate_test_train()
# AllKFDataset.save_test_train_h5("AllKFDatasets9K/")

Dataset = TrackDataSet("TTTrackDataset_TT12")
Dataset.load_data_from_root("/home/cb719/Documents/DataSets/VertexDatasets/OldKFGTTData_Old12/GTT_TrackNtuple_TT12.root",297676)
Dataset.generate_test_train()
Dataset.save_test_train_h5("FullTT12/")
'''
#######################################################################################################
# 1 KF TTTrack
# 2 KF Track
# 3 KF Track + Stubs
# 4 KF Track + Chi
# 5 KF Track + Matrix
# 6 KF Track + Chi + Matrix
# 7 KF Track + Stubs + Chi
# 8 KF Track + Stubs + Matrix
# 9 KF Track + Stubs + Chi + Matrix
#######################################################################################################
# Generate Base Dataset with all features
# 1 KF TTTrack
TTTrackDataset = TrackDataSet("TTTrackDataset_9K")
TTTrackDataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
TTTrackDataset.generate_test_train()
TTTrackDataset.save_test_train_h5("TTTrackDatasets9K/")
#######################################################################################################
# 9 KF Track + Stubs + Chi + Matrix
AllKFDataset = KFDataSet("KFDataset_WithAll_9K",withstubs=True,withchi=True,withmatrix=True)
AllKFDataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
AllKFDataset.generate_test_train()
AllKFDataset.save_test_train_h5("AllKFDatasets9K/")
#######################################################################################################
# 2 KF Track 
TrackKFDataset = KFDataSet("TrackKF_Dataset_9K",orig=AllKFDataset)
TrackKFDataset.training_features = ["b_trk_inv2R","b_trk_cot","b_trk_zT","b_trk_phiT"]
TrackKFDataset.generate_test_train()
TrackKFDataset.save_test_train_h5("TrackKFDatasets9K/")
#######################################################################################################
# 3 KF Track + Stubs
TrackKF_Stubs_Dataset= KFDataSet("TrackKF_Stubs_Dataset_9K",orig=AllKFDataset)
TrackKF_Stubs_Dataset.training_features = ["b_trk_inv2R","b_trk_cot","b_trk_zT","b_trk_phiT",
                                           "b_stub_r_1","b_stub_phi_1","b_stub_z_1","b_stub_dPhi_1","b_stub_dZ_1","b_stub_layer_1",
                                           "b_stub_r_2","b_stub_phi_2","b_stub_z_2","b_stub_dPhi_2","b_stub_dZ_2","b_stub_layer_2",
                                           "b_stub_r_3","b_stub_phi_3","b_stub_z_3","b_stub_dPhi_3","b_stub_dZ_3","b_stub_layer_3",
                                           "b_stub_r_4","b_stub_phi_4","b_stub_z_4","b_stub_dPhi_4","b_stub_dZ_4","b_stub_layer_4"]
TrackKF_Stubs_Dataset.generate_test_train()
TrackKF_Stubs_Dataset.save_test_train_h5("TrackKF_Stubs_Datasets9K/")
#######################################################################################################
# 4 KF Track + Chi
TrackKF_Chi_Dataset = KFDataSet("TrackKF_Chi_Dataset_9K",orig=AllKFDataset)
TrackKF_Chi_Dataset.training_features = ["b_trk_inv2R","b_trk_cot","b_trk_zT","b_trk_phiT",
                                         "bit_bendchi2","bit_chi2rz","bit_chi2rphi"]
TrackKF_Chi_Dataset.generate_test_train()
TrackKF_Chi_Dataset.save_test_train_h5("TrackKF_Chi_Datasets9K/")
#######################################################################################################
# 5 KF Track + Matrix
TrackKF_Matrix_Dataset = KFDataSet("TrackKF_Matrix_Dataset_9K",orig=AllKFDataset)
TrackKF_Matrix_Dataset.training_features = ["b_trk_inv2R","b_trk_cot","b_trk_zT","b_trk_phiT",
                                            "bit_C00", "bit_C01","bit_C11","bit_C22", "bit_C23","bit_C33"]
TrackKF_Matrix_Dataset.generate_test_train()
TrackKF_Matrix_Dataset.save_test_train_h5("TrackKF_Matrix_Datasets9K/")
#######################################################################################################
# 6 KF Track + Matrix + Chi
TrackKF_Matrix_Chi_Dataset = KFDataSet("TrackKF_Matrix_Chi_Dataset_9K",orig=AllKFDataset)
TrackKF_Matrix_Chi_Dataset.training_features = ["b_trk_inv2R","b_trk_cot","b_trk_zT","b_trk_phiT",
                                                "bit_bendchi2","bit_chi2rz","bit_chi2rphi",
                                                "bit_C00", "bit_C01","bit_C11","bit_C22", "bit_C23","bit_C33"]
TrackKF_Matrix_Chi_Dataset.generate_test_train()
TrackKF_Matrix_Chi_Dataset.save_test_train_h5("TrackKF_Matrix_Chi_Datasets9K/")
#######################################################################################################
# 7 KF Track + Stubs + Chi
TrackKF_Stubs_Chi_Dataset = KFDataSet("TrackKF_Stubs_Chi_Dataset_9K",orig=AllKFDataset)
TrackKF_Stubs_Chi_Dataset.training_features = ["b_trk_inv2R","b_trk_cot","b_trk_zT","b_trk_phiT",
                                               "b_stub_r_1","b_stub_phi_1","b_stub_z_1","b_stub_dPhi_1","b_stub_dZ_1","b_stub_layer_1",
                                               "b_stub_r_2","b_stub_phi_2","b_stub_z_2","b_stub_dPhi_2","b_stub_dZ_2","b_stub_layer_2",
                                               "b_stub_r_3","b_stub_phi_3","b_stub_z_3","b_stub_dPhi_3","b_stub_dZ_3","b_stub_layer_3",
                                               "b_stub_r_4","b_stub_phi_4","b_stub_z_4","b_stub_dPhi_4","b_stub_dZ_4","b_stub_layer_4",
                                               "bit_bendchi2","bit_chi2rz","bit_chi2rphi"]
TrackKF_Stubs_Chi_Dataset.generate_test_train()
TrackKF_Stubs_Chi_Dataset.save_test_train_h5("TrackKF_Stubs_Chi_Datasets9K/")
#######################################################################################################
# 8 KF Track + Stubs + Matrix
TrackKF_Stubs_Matrix_Dataset = KFDataSet("TrackKF_Stubs_Matrix_Dataset_9K",orig=AllKFDataset)
TrackKF_Stubs_Matrix_Dataset.training_features = ["b_trk_inv2R","b_trk_cot","b_trk_zT","b_trk_phiT",
                                                  "b_stub_r_1","b_stub_phi_1","b_stub_z_1","b_stub_dPhi_1","b_stub_dZ_1","b_stub_layer_1",
                                                  "b_stub_r_2","b_stub_phi_2","b_stub_z_2","b_stub_dPhi_2","b_stub_dZ_2","b_stub_layer_2",
                                                  "b_stub_r_3","b_stub_phi_3","b_stub_z_3","b_stub_dPhi_3","b_stub_dZ_3","b_stub_layer_3",
                                                  "b_stub_r_4","b_stub_phi_4","b_stub_z_4","b_stub_dPhi_4","b_stub_dZ_4","b_stub_layer_4",
                                                  "bit_C00", "bit_C01","bit_C11","bit_C22", "bit_C23","bit_C33"]
TrackKF_Stubs_Matrix_Dataset.generate_test_train()
TrackKF_Stubs_Matrix_Dataset.save_test_train_h5("TrackKF_Stubs_Matrix_Datasets9K/")
'''

# 10 Track w/o bendchi
# Track_nobendChi_Dataset = TrackDataSet("Track_nobendChi_Dataset_9K")
# Track_nobendChi_Dataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
# Track_nobendChi_Dataset.training_features = [  "bit_chi2rphi",
#                                                "bit_chi2rz",
#                                                "bit_TanL",
#                                                "bit_z0",
#                                                "bit_InvR",
#                                                "pred_nstub",
#                                                "nlay_miss"]
# Track_nobendChi_Dataset.generate_test_train()
# Track_nobendChi_Dataset.save_test_train_h5("Track_nobendChi_Datasets9K/")
# # 1 KF TTTrack

# TTTrackxgboostmodel = XGBoostClassifierModel()
# TTTrackxgboostmodel.load_data("TTTrackDatasets9K/")
# TTTrackxgboostmodel.train()
# TTTrackxgboostmodel.save_model("Models/TTTrack")
# TTTrackxgboostmodel.test()
# TTTrackxgboostmodel.evaluate(plot=True,name="TTTrack Parameters")
# TTTrackxgboostmodel.plot_model()
# TTTrackxgboostmodel.ONNX_convert_model("Models_12/TTTrack")

# # KF TTTrack No Bend

# TTTracknbxgboostmodel = XGBoostClassifierModel()
# TTTracknbxgboostmodel.load_data("Track_nobendChi_Datasets9K/")
# TTTracknbxgboostmodel.train()
# TTTracknbxgboostmodel.save_model("Models/TTTrack_nobend")
# TTTracknbxgboostmodel.test()
# TTTracknbxgboostmodel.evaluate(plot=True,name="TTTrack No Bend")
# TTTracknbxgboostmodel.plot_model()
# TTTracknbxgboostmodel.ONNX_convert_model("Models_12/TTTrack_nobend")

# #######################################################################################################
# # 2 KF Track

# KFTrackxgboostmodel = XGBoostClassifierModel()
# KFTrackxgboostmodel.load_data("TrackKFDatasets9K/")
# KFTrackxgboostmodel.train()
# KFTrackxgboostmodel.save_model("Models/KFTrack")
# KFTrackxgboostmodel.test()
# KFTrackxgboostmodel.evaluate(plot=True,name="KF Track ")
# KFTrackxgboostmodel.plot_model()
# KFTrackxgboostmodel.ONNX_convert_model("Models_12/KFTrack")
# # #######################################################################################################

# # 3 KF Track + Stubs

# KFTrack_Stubs_xgboostmodel = XGBoostClassifierModel()
# KFTrack_Stubs_xgboostmodel.load_data("TrackKF_Stubs_Datasets9K/")
# KFTrack_Stubs_xgboostmodel.train()
# KFTrack_Stubs_xgboostmodel.save_model("Models/KFTrack_Stubs")
# KFTrack_Stubs_xgboostmodel.test()
# KFTrack_Stubs_xgboostmodel.evaluate(plot=True,name="KF Track + Stubs")
# KFTrack_Stubs_xgboostmodel.plot_model()
# KFTrack_Stubs_xgboostmodel.ONNX_convert_model("Models_12/KFTrack_Stubs")
# #######################################################################################################

# # 4 KF Track + Chi

# KFTrack_Chi_xgboostmodel = XGBoostClassifierModel()
# KFTrack_Chi_xgboostmodel.load_data("TrackKF_Chi_Datasets9K/")
# KFTrack_Chi_xgboostmodel.train()
# KFTrack_Chi_xgboostmodel.save_model("Models/KFTrack_Chi")
# KFTrack_Chi_xgboostmodel.test()
# KFTrack_Chi_xgboostmodel.evaluate(plot=True,name="KF Track + Chi")
# KFTrack_Chi_xgboostmodel.plot_model()
# KFTrack_Chi_xgboostmodel.ONNX_convert_model("Models_12/KFTrack_Chi")
# #######################################################################################################
# # 5 KF Track + Matrix

# KFTrack_Matrix_xgboostmodel = XGBoostClassifierModel()
# KFTrack_Matrix_xgboostmodel.load_data("TrackKF_Matrix_Datasets9K/")
# KFTrack_Matrix_xgboostmodel.train()
# KFTrack_Matrix_xgboostmodel.save_model("Models/KFTrack_Matrix")
# KFTrack_Matrix_xgboostmodel.test()
# KFTrack_Matrix_xgboostmodel.evaluate(plot=True,name="KF Track + Matrix")
# KFTrack_Matrix_xgboostmodel.plot_model()
# KFTrack_Matrix_xgboostmodel.ONNX_convert_model("Models_12/KFTrack_Matrix")
# #######################################################################################################
# # 6 KF Track + Chi + Matrix

# KFTrack_Matrix_Chi_xgboostmodel = XGBoostClassifierModel()
# KFTrack_Matrix_Chi_xgboostmodel.load_data("TrackKF_Matrix_Chi_Datasets9K/")
# KFTrack_Matrix_Chi_xgboostmodel.train()
# KFTrack_Matrix_Chi_xgboostmodel.save_model("Models/KFTrack_Matrix_Chi")
# KFTrack_Matrix_Chi_xgboostmodel.test()
# KFTrack_Matrix_Chi_xgboostmodel.evaluate(plot=True,name="KF Track + Matrix + Chi")
# KFTrack_Matrix_Chi_xgboostmodel.plot_model()
# KFTrack_Matrix_Chi_xgboostmodel.ONNX_convert_model("Models_12/KFTrack_Matrix_Chi")
# #######################################################################################################
# # 7 KF Track + Stubs + Chi

# KFTrack_Stubs_Chi_xgboostmodel = XGBoostClassifierModel()
# KFTrack_Stubs_Chi_xgboostmodel.load_data("TrackKF_Stubs_Chi_Datasets9K/")
# KFTrack_Stubs_Chi_xgboostmodel.train()
# KFTrack_Stubs_Chi_xgboostmodel.save_model("Models/KFTrack_Stubs_Chi")
# KFTrack_Stubs_Chi_xgboostmodel.test()
# KFTrack_Stubs_Chi_xgboostmodel.evaluate(plot=True,name="KF Track + Stubs + Chi")
# KFTrack_Stubs_Chi_xgboostmodel.plot_model()
# KFTrack_Stubs_Chi_xgboostmodel.ONNX_convert_model("Models_12/KFTrack_Stubs_Chi")
# #######################################################################################################
# # 8 KF Track + Stubs + Matrix

# KFTrack_Stubs_Matrix_xgboostmodel = XGBoostClassifierModel()
# KFTrack_Stubs_Matrix_xgboostmodel.load_data("TrackKF_Stubs_Matrix_Datasets9K/")
# KFTrack_Stubs_Matrix_xgboostmodel.train()
# KFTrack_Stubs_Matrix_xgboostmodel.save_model("Models/KFTrack_Stubs_Matrix")
# KFTrack_Stubs_Matrix_xgboostmodel.test()
# KFTrack_Stubs_Matrix_xgboostmodel.evaluate(plot=True,name="KF Track + Stubs + Matrix")
# KFTrack_Stubs_Matrix_xgboostmodel.plot_model()
# KFTrack_Stubs_Matrix_xgboostmodel.ONNX_convert_model("Models_12/KFTrack_Stubs_Matrix")
# #######################################################################################################
# # 9 KF Track + Stubs + Chi + Matrix

# KFTrack_Stubs_Matrix_Chi_xgboostmodel = XGBoostClassifierModel()
# KFTrack_Stubs_Matrix_Chi_xgboostmodel.load_data("AllKFDatasets9K/")
# KFTrack_Stubs_Matrix_Chi_xgboostmodel.train()
# KFTrack_Stubs_Matrix_Chi_xgboostmodel.save_model("Models/KFTrack_Stubs_Matrix_Chi")
# KFTrack_Stubs_Matrix_Chi_xgboostmodel.test()
# KFTrack_Stubs_Matrix_Chi_xgboostmodel.evaluate(plot=True,name="KF Track + Stubs + Matrix + Chi")
# KFTrack_Stubs_Matrix_Chi_xgboostmodel.plot_model()
# KFTrack_Stubs_Matrix_Chi_xgboostmodel.ONNX_convert_model("Models_12/KFTrack_Stubs_Matrix_Chi")
# #######################################################################################################
# '''

#NewKFTrackxgboostmodel = XGBoostClassifierModel()
#NewKFTrackxgboostmodel.load_data("NewKFTrackDatasets/")
#NewKFTrackxgboostmodel.comet_project_name = "Xgboost_newKFTrack"
#NewKFTrackxgboostmodel.min_child_weight =  {"min":0,"max":10, "value":7.520001241928147}
#NewKFTrackxgboostmodel.alpha            =  {"min":0,"max":1,  "value":0.8014407138031248}
#NewKFTrackxgboostmodel.early_stopping   =  {"min":1,"max":20, "value":5}
#NewKFTrackxgboostmodel.learning_rate    =  {"min":0,"max":1,   "value":0.8221936262293147	}
#NewKFTrackxgboostmodel.n_estimators     =  {"min":0,"max":100,   "value":100}
#NewKFTrackxgboostmodel.subsample        =  {"min":0,"max":0.99,"value":0.5123763510523847	}
#NewKFTrackxgboostmodel.max_depth        =  {"min":1,"max":3  ,"value":3 }
#NewKFTrackxgboostmodel.gamma            =  {"min":0,"max":0.99,"value":	0.38212524090081945}
#NewKFTrackxgboostmodel.rate_drop        =  {"min":0,"max":1,"value":0.788588}
#NewKFTrackxgboostmodel.skip_drop        =  {"min":0,"max":1,"value":0.147907}
#NewKFTrackxgboostmodel.optimise()
#NewKFTrackxgboostmodel.train()
#NewKFTrackxgboostmodel.save_model("Models/NewKFTrack")
#NewKFTrackxgboostmodel.load_model("Models/NewKFTrack")
#NewKFTrackxgboostmodel.test()
#for i,item in enumerate(NewKFTrackxgboostmodel.y_predict_proba):
#    print(NewKFTrackxgboostmodel.DataSet.X_test.iloc[i].tolist())
#    print("prediction:",item)
#NewKFTrackxgboostmodel.evaluate(plot=True,name="New KF Track")
#NewKFTrackxgboostmodel.plot_model()
#NewKFTrackxgboostmodel.ONNX_convert_model("Models/NewKFTrack")
#NewKFTrackxgboostmodel.synth_model(hls=False,hdl=False,intwidth=24,fracwidth=16,plot=True,test_events=100000)

#NewKFKFxgboostmodel = XGBoostClassifierModel()
#NewKFKFxgboostmodel.load_data("NewKFKFDatasets/")
#NewKFKFxgboostmodel.comet_project_name = "Xgboost_newKF"
#NewKFKFxgboostmodel.min_child_weight =  {"min":0,"max":10, "value":7.554038941635881}
#NewKFKFxgboostmodel.alpha            =  {"min":0,"max":1,  "value":0.1455098913001549}
#NewKFKFxgboostmodel.early_stopping   =  {"min":1,"max":20, "value":5}
#NewKFKFxgboostmodel.learning_rate    =  {"min":0,"max":1,   "value":0.9513310026293323	}
#NewKFKFxgboostmodel.n_estimators     =  {"min":0,"max":100,   "value":74}
#NewKFKFxgboostmodel.subsample        =  {"min":0,"max":0.99,"value":0.9171935681586878	}
#NewKFKFxgboostmodel.max_depth        =  {"min":1,"max":3  ,"value":2 }
#NewKFKFxgboostmodel.gamma            =  {"min":0,"max":0.99,"value":	0.3653854859526517}
#NewKFKFxgboostmodel.rate_drop        =  {"min":0,"max":1,"value":0.788588}
#NewKFKFxgboostmodel.skip_drop        =  {"min":0,"max":1,"value":0.147907}
#NewKFKFxgboostmodel.optimise()
#NewKFKFxgboostmodel.train()#
#NewKFKFxgboostmodel.save_model("Models/NewKFKF")
#NewKFKFxgboostmodel.load_model("Models/NewKFKF")
#NewKFKFxgboostmodel.test()
#for i,item in enumerate(NewKFKFxgboostmodel.y_predict_proba):
#    print(NewKFKFxgboostmodel.DataSet.X_test.iloc[i].tolist())
 #   print("prediction:",item)

#NewKFKFxgboostmodel.evaluate(plot=True,name="New KF KF")
#NewKFKFxgboostmodel.plot_model()

#NewKFKFxgboostmodel.ONNX_convert_model("Models/NewKFKF")
#NewKFTrackxgboostmodel.synth_model(hls=False,hdl=False,intwidth=24,fracwidth=16,plot=True,test_events=100000)


OldKFTrackxgboostmodel = XGBoostClassifierModel()
OldKFTrackxgboostmodel.load_data("FullTT12/")
#OldKFTrackxgboostmodel.comet_project_name = "Xgboost_oldKF"
#OldKFTrackxgboostmodel.optimise()
#OldKFTrackxgboostmodel.min_child_weight =  {"min":0,"max":10, "value":8.848520598251687	}
#OldKFTrackxgboostmodel.alpha            =  {"min":0,"max":1,  "value":0.2923963839055528 }
#OldKFTrackxgboostmodel.early_stopping   =  {"min":1,"max":20, "value":5}
#OldKFTrackxgboostmodel.learning_rate    =  {"min":0,"max":1,   "value":0.61217681482078}
#OldKFTrackxgboostmodel.n_estimators     =  {"min":0,"max":100,   "value":83}
#OldKFTrackxgboostmodel.subsample        =  {"min":0,"max":0.99,"value":0.38499426500459616 }
#OldKFTrackxgboostmodel.max_depth        =  {"min":1,"max":3  ,"value":2 }
#OldKFTrackxgboostmodel.gamma            =  {"min":0,"max":0.99,"value":	0.4848495035336481 }
#OldKFTrackxgboostmodel.rate_drop        =  {"min":0,"max":1,"value":0.788588}
#OldKFTrackxgboostmodel.skip_drop        =  {"min":0,"max":1,"value":0.147907}
OldKFTrackxgboostmodel.train()#
OldKFTrackxgboostmodel.save_model("Models/OldKFTrack")
#OldKFTrackxgboostmodel.load_model("Models/OldKFTrack")
OldKFTrackxgboostmodel.test()
#for i,item in enumerate(OldKFTrackxgboostmodel.y_predict_proba):
#    print(OldKFTrackxgboostmodel.DataSet.X_test.iloc[i].tolist())
#    print("prediction:",item)
OldKFTrackxgboostmodel.evaluate(plot=True,name="Old KF Track")
OldKFTrackxgboostmodel.ONNX_convert_model("Models/OldKFTrack")
OldKFTrackxgboostmodel.plot_model()

OldKFChi2Model = CutClassifierModel()
OldKFChi2Model.load_data("OldKFFloatingTrackdatasets/")
OldKFChi2Model.test()
OldKFChi2Model.evaluate(plot=True,name="Old KF Track chi2")


# NewKFChi2Model = CutClassifierModel()
# NewKFChi2Model.load_data("NewKFFloatingTrackDatasets/")
# NewKFChi2Model.test()
# NewKFChi2Model.evaluate(plot=True,name="New KF Track chi2")
# '''