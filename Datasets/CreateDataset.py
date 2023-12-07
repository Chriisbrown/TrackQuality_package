from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

# Dataset = TrackDataSet("MCDegredation_5")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/MCDegredation/LargeSamples/MC_Degredation5_TT_TrackNtuple.root",200000)#297676)
# Dataset.generate_test_train()
# Dataset.save_test_train_h5("MCDegredation_5/")

# Dataset = TrackDataSet("Degredation1_Train")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation1/Degredation1_Train_TrackNtuple.root",300000)#297676)
# Dataset.generate_test()
# Dataset.save_test_train_h5("Degredation1_Train/")

# Dataset = TrackDataSet("NewKF_NoDegredation_Zp")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/NoDegredation/NewKF_NoDegredation_Zp_TrackNtuple.root",300000)#297676)
# Dataset.generate_test()
# Dataset.save_test_train_h5("NewKF_NoDegredation_Zp/")

# Dataset = TrackDataSet("NewKF_Degredation9_Zp")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation9/NewKF_Degredation9_Zp_TrackNtuple.root",300000)#297676)
# Dataset.generate_test()
# Dataset.save_test_train_h5("NewKF_Degredation9_Zp/")

# Dataset = TrackDataSet("NewKF_NoDegredation_Train")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/NoDegredation/NewKF_NoDegredation_Train_TrackNtuple.root",300000)#297676)
# Dataset.generate_train()
# Dataset.save_test_train_h5("NewKF_NoDegredation_Train/")

# Dataset = TrackDataSet("NewKF_Degredation9_Train")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation9/NewKF_Degredation9_Train_TrackNtuple.root",300000)#297676)
# Dataset.generate_train()
# Dataset.save_test_train_h5("NewKF_Degredation9_Train/")

# Dataset = TrackDataSet("NewKF_Degredation9_1")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation9/NewKF_Degredation9_1_TrackNtuple.root",300000)#297676)
# Dataset.generate_train()
# Dataset.save_test_train_h5("NewKF_Degredation9_1/")

Dataset = TrackDataSet("Degredation3_Zp")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation3/Degredation3_Zp_TrackNtuple.root",300000)#297676)
Dataset.generate_test()
Dataset.save_test_train_h5("Degredation3_Zp/")

Dataset = TrackDataSet("Degredation3_Train")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation3/Degredation3_Train_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("Degredation3_Train/")

Dataset = TrackDataSet("Degredation3_1")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degredation3/Degredation3_1_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("Degredation3_1/")