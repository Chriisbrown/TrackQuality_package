from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import Binned_CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Train", help = "Train Folder")
parser.add_argument("--Test", help = "Test Folder")
parser.add_argument("--Model", help = "Model file")
args = parser.parse_args()

setmatplotlib()

train_folder = args.Train
test_folder = args.Test
model_folder = args.Model
name = train_folder.split("_")[0]

plot_types = ["ROC","FPR","TPR","score"]
    
model = XGBoostClassifierModel(name)
if args.Train is not None:
    if not os.path.exists('Projects/'+train_folder):
        os.system("mkdir Projects/"+train_folder)
        os.system("mkdir Projects/"+train_folder+"/Plots")
        os.system("mkdir Projects/"+train_folder+"/FW")
        os.system("mkdir Projects/"+train_folder+"/Models")
    
    model.load_data("Datasets/"+train_folder+"/")
    model.train()
    model.save_model("Projects/"+train_folder+"/Models/",train_folder+"_XGB")
    model.load_model("Projects/"+train_folder+"/Models/",train_folder+"_XGB")
elif args.Model is not None:
    model.load_model("Projects/"+model_folder+"/Models/",model_folder+"_XGB")

if args.Test is not None:
    if not os.path.exists('Projects/'+test_folder):
        os.system("mkdir Projects/"+test_folder)
        os.system("mkdir Projects/"+test_folder+"/Plots")
        os.system("mkdir Projects/"+test_folder+"/FW")
        os.system("mkdir Projects/"+test_folder+"/Models")
    model.load_data("Datasets/"+test_folder+"/")
    model.test()
    model.evaluate(plot=True,save_dir="Projects/"+test_folder+"/Plots/")
    model.full_save("Projects/"+test_folder+"/Models/"+test_folder+"/",test_folder+"_XGB")
    model.full_load("Projects/"+test_folder+"/Models/"+test_folder+"/",test_folder+"_XGB")
    plot_model(model,"Projects/"+test_folder+"/")

