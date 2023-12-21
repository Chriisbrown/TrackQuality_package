from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

#### No Degradation Datasets #####

Dataset = TrackDataSet("Degradation0_Test")
Dataset.load_data_from_root("/eos/user/c/cebrown/DegradationStudies/Degradation0/Degradation0_Test_TrackNtuple.root",300000)#297676)
Dataset.generate_test()
Dataset.save_test_train_h5("/eos/user/k/klaw/ML_L1_workshop/TrackQuality_package/Datasets_with_weights/Degradation0_Test/")

Dataset = TrackDataSet("Degradation0_Train")
Dataset.load_data_from_root("/eos/user/c/cebrown/DegradationStudies/Degradation0/Degradation0_Train_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("/eos/user/k/klaw/ML_L1_workshop/TrackQuality_package/Datasets_with_weights/Degradation0_Train/")

Dataset = TrackDataSet("Degradation0_1")
Dataset.load_data_from_root("/eos/user/c/cebrown/DegradationStudies/Degradation0/Degradation0_1_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("/eos/user/k/klaw/ML_L1_workshop/TrackQuality_package/Datasets_with_weights/Degradation0_1/")

'''
#### 1% Bad Stubs Datasets #####

Dataset = TrackDataSet("Degradation1_Test")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation1/Degradation1_Test_TrackNtuple.root",300000)#297676)
Dataset.generate_test()
Dataset.save_test_train_h5("Degradation1_Test/")

Dataset = TrackDataSet("Degradation1_Train")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation1/Degradation1_Train_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("Degradation1_Train/")

Dataset = TrackDataSet("Degradation1_1")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation1/Degradation1_1_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("Degradation1_1/")

#### 5% Bad Stubs Datasets #####

Dataset = TrackDataSet("Degradation5_Test")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation5/Degradation5_Test_TrackNtuple.root",300000)#297676)
Dataset.generate_test()
Dataset.save_test_train_h5("Degradation5_Test/")

Dataset = TrackDataSet("Degradation5_Train")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation5/Degradation5_Train_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("Degradation5_Train/")

Dataset = TrackDataSet("Degradation5_1")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation5/Degradation5_1_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("Degradation5_1/")

#### 10% Bad Stubs Datasets #####

Dataset = TrackDataSet("Degradation10_Test")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation10/Degradation10_Test_TrackNtuple.root",300000)#297676)
Dataset.generate_test()
Dataset.save_test_train_h5("Degradation10_Test/")

Dataset = TrackDataSet("Degradation10_Train")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation10/Degradation10_Train_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("Degradation10_Train/")

Dataset = TrackDataSet("Degradation10_1")
Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/Degradation10/Degradation10_1_TrackNtuple.root",300000)#297676)
Dataset.generate_train()
Dataset.save_test_train_h5("Degradation10_1/")
'''
