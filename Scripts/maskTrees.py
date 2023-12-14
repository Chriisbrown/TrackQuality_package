from Models.GBDTTrackQualityModel import XGBoostClassifierModel
from Datasets.Dataset import *
from Utils.util import *
from Utils.TreeVis import *
import numpy as np
from scipy.special import expit
import os
import matplotlib.animation as animation
model = XGBoostClassifierModel("Degredation_0")
filepath = "/home/ryanm124/ml_hackathon/TrackQuality_package/Scripts/Projects/Retrained/Models/"
model.load_model(filepath,"Retrained01_XGB")
cut_value = 1000.0

booster = model.model.get_booster()
treedict = convert(booster)
trees = [_ for _ in booster]
for i in range(treedict['n_trees']):
#    print("Tree #: ",i)
#    print(treedict['trees'][i][0]['value'][0])
    if treedict['trees'][i][0]['value'][0]<cut_value:
        booster = booster[:i]
        break
model.model._Booster = booster
model.save_model(filepath,"Retrained01Trim_XGB")
