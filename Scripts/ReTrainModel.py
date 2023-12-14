from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os




RetrainModel = "Retrained015"
RetrainFolder = "Retrained"


model = XGBoostClassifierModel("Degredation_0")
#filepath = "/home/ryanm124/ml_hackathon/TrackQuality_package/Projects/Degradation0_Test/Models/Degradation0/"
filepath = "/home/ryanm124/ml_hackathon/TrackQuality_package/Scripts/Projects/Retrained/Models/" 
model.load_model(filepath,"Retrained01Trim_XGB")


datapath = "/home/ryanm124/ml_hackathon/TrackQuality_package/Datasets/Degradation5_Train/"
model.load_data(datapath)
model.retrain()

model.save_model("Projects/"+RetrainFolder+"/Models/",RetrainModel+"_XGB")
model.load_model("Projects/"+RetrainFolder+"/Models/",RetrainModel+"_XGB")

model.load_data("/home/ryanm124/ml_hackathon/TrackQuality_package/Datasets/Degradation10_Test/")
plot_model(model,"Projects/"+RetrainFolder+"/")
model.test()
model.evaluate(plot=True,save_dir="Projects/"+RetrainFolder+"/Plots/")
