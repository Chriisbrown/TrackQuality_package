from GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from CutTrackQualityModel import CutClassifierModel
from Dataset import *
import numpy as np
import os
import tensorflow_decision_forests as tfdf

#Dataset = TrackDataSet("NoDegredation_Train")
#Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/NoDegredation/NoDegredation_Train_TrackNtuple.root",200000)#297676)
#Dataset.generate_test_train()
#Dataset.save_test_train_h5("NoDegredation_Train/")

# Dataset = TrackDataSet("NoDegredation_Zp")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/NoDegredation/NoDegredation_Zp_TrackNtuple.root",300000)#297676)
# Dataset.generate_test()
# Dataset.save_test_train_h5("NoDegredation_Zp/")

NoDegredation = XGBoostClassifierModel()
NoDegredation.load_data("NoDegredation_Train/")
#NoDegredation.train()#
#NoDegredation.save_model("Models/NoDegredation")
NoDegredation.load_model("Models/NoDegredation")
NoDegredation.test()
NoDegredation.evaluate(plot=True,name="TTtest")
NoDegredation.plot_model()
NoDegredation.synth_model(sim=True,hdl=True,hls=False,cpp=False,onnx=False)

