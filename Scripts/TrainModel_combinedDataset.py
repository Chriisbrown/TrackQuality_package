from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import Binned_CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

setmatplotlib()

plot_types = ["ROC","FPR","TPR","score"]

folder = "CombiModel"

os.system("mkdir Projects/"+folder)
os.system("mkdir Projects/"+folder+"/Plots")
os.system("mkdir Projects/"+folder+"/FW")
os.system("mkdir Projects/"+folder+"/Models")

#Create 4 smaller sub datasets, each one from a different sub sample root file to prevent duplicate events in the training 

# subdataset0 = TrackDataSet("Degradation0_1")
# subdataset0.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation0/Degradation0_1_TrackNtuple.root",300000)#297676)
# subdataset0.generate_train()
# subdataset0.save_test_train_h5("Degradation0_1/")
subdataset0 = DataSet.fromTrainTest("Datasets/Degradation0/Degradation0_Train/")

# subdataset1 = TrackDataSet("Degradation1_2")
# subdataset1.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation1/Degradation1_2_TrackNtuple.root",300000)#297676)
# subdataset1.generate_train()
# subdataset1.save_test_train_h5("Degradation1_2/")
subdataset1 = DataSet.fromTrainTest("Datasets/Degradation1/Degradation1_Train/")

# subdataset5 = TrackDataSet("Degradation5_3")
# subdataset5.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation5/Degradation5_3_TrackNtuple.root",300000)#297676)
# subdataset5.generate_train()
# subdataset5.save_test_train_h5("Degradation5_3/")
subdataset5 = DataSet.fromTrainTest("Datasets/Degradation5/Degradation5_Train/")

# subdataset10 = TrackDataSet("Degradation10_4")
# subdataset10.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation10/Degradation10_4_TrackNtuple.root",300000)#297676)
# subdataset10.generate_train()
# subdataset10.save_test_train_h5("Degradation10_4/")
subdataset10 = DataSet.fromTrainTest("Datasets/Degradation10/Degradation10_Train/")
model = XGBoostClassifierModel(folder)

model.DataSet = subdataset0 + subdataset1 + subdataset5 + subdataset10
model.train()
model.save_model("Projects/"+folder+"/Models/",folder+"_XGB")
model.load_model("Projects/"+folder+"/Models/",folder+"_XGB")

model.load_data("Datasets/Degradation10/Degradation10_Test/")
plot_model(model,"Projects/"+folder+"/")
model.test()
model.evaluate(plot=True,save_dir="Projects/"+folder+"/Plots/")

    
