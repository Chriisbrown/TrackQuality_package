from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os


folder_list = ["Degradation0","Degradation1","Degradation5","Degradation10"]

RetrainModel = "Retrained"
RetrainFolder = "Retrained"


model = XGBoostClassifierModel(folder_list[0])
model.load_data("Datasets/"+folder_list[0]+"/"+folder_list[0]+"_Train/")
model.n_estimators = 15
model.train()
folder_list.pop(0)

for i,folder in enumerate(folder_list):
    model.load_data("Datasets/"+folder+"/"+folder+"_Train/")
    model.retrain()

model.save_model("Projects/"+RetrainFolder+"/Models/",RetrainModel+"_XGB")
model.load_model("Projects/"+RetrainFolder+"/Models/",RetrainModel+"_XGB")

model.load_data("Datasets/Degradation10/Degradation10_Test/")
plot_model(model,"Projects/"+RetrainFolder+"/")
model.test()
model.evaluate(plot=True,save_dir="Projects/"+RetrainFolder+"/Plots/")
