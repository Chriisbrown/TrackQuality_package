from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

Dataset = TrackDataSet("MCDegredation_5")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/MCDegredation/LargeSamples/MC_Degredation5_TT_TrackNtuple.root",200000)#297676)
Dataset.generate_test_train()
Dataset.save_test_train_h5("MCDegredation_5/")

Dataset = TrackDataSet("Degredation1_Train")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation1/Degredation1_Train_TrackNtuple.root",300000)#297676)
Dataset.generate_test()
Dataset.save_test_train_h5("Degredation1_Train/")

Dataset = TrackDataSet("Degredation1_Zp")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation1/Degredation1_Zp_TrackNtuple.root",300000)#297676)
Dataset.generate_test()
Dataset.save_test_train_h5("Degredation1_Zp/")