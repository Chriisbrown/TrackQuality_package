from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os
from Utils.TreeVis import *



RetrainModel = "Retrained0123"
RetrainFolder = "Retrained"


model = XGBoostClassifierModel("Degredation_0")
#filepath = "/home/ryanm124/ml_hackathon/TrackQuality_package/Projects/Degradation0_Test/Models/Degradation0/"
filepath = "/home/ryanm124/ml_hackathon/TrackQuality_package/Scripts/Projects/Retrained/Models/" 
model.load_model(filepath,"Retrained012Trim_XGB")
booster = model.model.get_booster()
treedict = convert(booster)
datapath = "/home/ryanm124/ml_hackathon/TrackQuality_package/Datasets/Degradation3_Train/"
model.load_data(datapath)
model.model.set_params(n_estimators=treedict["n_trees"])
print(model.model.get_params())
print("trees to train",60-treedict["n_trees"])
