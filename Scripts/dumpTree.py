from Models.GBDTTrackQualityModel import XGBoostClassifierModel
from Datasets.Dataset import *
from Utils.util import *
from Utils.TreeVis import *
import numpy as np
from scipy.special import expit
import os
import matplotlib.animation as animation
model = XGBoostClassifierModel("Degredation_0")
model.load_model("/home/ryanm124/ml_hackathon/TrackQuality_package/Scripts/Projects/Retrained/Models/","Retrained01Trim_XGB")
treedict = convert(model.model.get_booster())
for i in range(treedict['n_trees']):
    print("Tree #: ",i)
    print(treedict['trees'][i][0]['value'][0])
