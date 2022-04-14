from GBDTTrackQualityModel import XGBoostClassifierModel,SklearnClassifierModel
from CutTrackQualityModel import CutClassifierModel
from Dataset import *
import numpy as np
import os


# Generate Datasets for all 4 models

newKFKFwMDataset = KFDataSet("NewKFKF_Dataset_WithMatrix_9K",False,True)
newKFKFwMDataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
newKFKFwMDataset.generate_test_train()
newKFKFwMDataset.save_test_train_h5("NewKFKFwMDatasets9K/")

newKFKFwCDataset = KFDataSet("NewKFKF_Dataset_WithChi_9K",True,False)
newKFKFwCDataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
newKFKFwCDataset.generate_test_train()
newKFKFwCDataset.save_test_train_h5("NewKFKFwCDatasets9K/")

newKFKFwMwCDataset = KFDataSet("NewKFKF_Dataset_WithChiWithMatrix_9K",True,True)
newKFKFwMwCDataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
newKFKFwMwCDataset.generate_test_train()
newKFKFwMwCDataset.save_test_train_h5("NewKFKFwMwCDatasets9K/")

newKFKFDataset = KFDataSet("NewKFKF_Dataset_9K",False,False)
newKFKFDataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
newKFKFDataset.generate_test_train()
newKFKFDataset.save_test_train_h5("NewKFKFDatasets9K/")

newKFTrackDataset = TrackDataSet("NewKFTrack_Dataset_9K")
newKFTrackDataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
newKFTrackDataset.generate_test_train()
newKFTrackDataset.save_test_train_h5("NewKFTrackDatasets9K/")

# Train Model with track features only

NewKFTrackxgboostmodel = XGBoostClassifierModel()
NewKFTrackxgboostmodel.load_data("NewKFTrackDatasets9K/")
NewKFTrackxgboostmodel.train()
NewKFTrackxgboostmodel.save_model("Models/NewKFTrack")
NewKFTrackxgboostmodel.test()
NewKFTrackxgboostmodel.evaluate(plot=True,name="New KF Track Parameters")
NewKFTrackxgboostmodel.plot_model()
NewKFTrackxgboostmodel.ONNX_convert_model("Models_12/NewKFTrack")

# Train Model with KF Track features and stubs

NewKFKFxgboostmodel = XGBoostClassifierModel()
NewKFKFxgboostmodel.load_data("NewKFKFDatasets9K/")
NewKFKFxgboostmodel.train()
NewKFKFxgboostmodel.save_model("Models/NewKFKF")
NewKFKFxgboostmodel.test()
NewKFKFxgboostmodel.evaluate(plot=True,name="New KF KF Parameters")
NewKFKFxgboostmodel.plot_model()
NewKFKFxgboostmodel.ONNX_convert_model("Models_12/NewKFKF")

# Train Model with KF Track features and stubs and chi

NewKFKFwCxgboostmodel = XGBoostClassifierModel()
NewKFKFwCxgboostmodel.load_data("NewKFKFwCDatasets9K/")
NewKFKFwCxgboostmodel.train()
NewKFKFwCxgboostmodel.save_model("Models/NewKFKFwC")
NewKFKFwCxgboostmodel.test()
NewKFKFwCxgboostmodel.evaluate(plot=True,name="New KF KF with Chi Parameters")
NewKFKFwCxgboostmodel.plot_model()
NewKFKFwCxgboostmodel.ONNX_convert_model("Models_12/NewKFKFwC")

# Train Model with KF Track features and stubs and matrix

NewKFKFwMxgboostmodel = XGBoostClassifierModel()
NewKFKFwMxgboostmodel.load_data("NewKFKFwMDatasets9K/")
NewKFKFwMxgboostmodel.train()
NewKFKFwMxgboostmodel.save_model("Models/NewKFKFwM")
NewKFKFwMxgboostmodel.test()
NewKFKFwMxgboostmodel.evaluate(plot=True,name="New KF KF with Matrix Parameters")
NewKFKFwMxgboostmodel.plot_model()
NewKFKFwMxgboostmodel.ONNX_convert_model("Models_12/NewKFKFwM")

# Train Model with KF Track features and stubs and matrix and chi

NewKFKFwMwCxgboostmodel = XGBoostClassifierModel()
NewKFKFwMwCxgboostmodel.load_data("NewKFKFwMwCDatasets9K/")
NewKFKFwMwCxgboostmodel.train()
NewKFKFwMwCxgboostmodel.save_model("Models/NewKFKFwMwC")
NewKFKFwMwCxgboostmodel.test()
NewKFKFwMwCxgboostmodel.evaluate(plot=True,name="New KF KF with Matrix and Chi Parameters")
NewKFKFwMwCxgboostmodel.plot_model()
NewKFKFwMwCxgboostmodel.ONNX_convert_model("Models_12/NewKFKFwMwC")


'''

NewKFTrackxgboostmodel = XGBoostClassifierModel()
NewKFTrackxgboostmodel.load_data("NewKFTrackDatasets/")
#NewKFTrackxgboostmodel.comet_project_name = "Xgboost_newKFTrack"
#NewKFTrackxgboostmodel.min_child_weight =  {"min":0,"max":10, "value":7.520001241928147}
#NewKFTrackxgboostmodel.alpha            =  {"min":0,"max":1,  "value":0.8014407138031248}
#NewKFTrackxgboostmodel.early_stopping   =  {"min":1,"max":20, "value":5}
#NewKFTrackxgboostmodel.learning_rate    =  {"min":0,"max":1,   "value":0.8221936262293147	}
#NewKFTrackxgboostmodel.n_estimators     =  {"min":0,"max":100,   "value":100}
#NewKFTrackxgboostmodel.subsample        =  {"min":0,"max":0.99,"value":0.5123763510523847	}
#NewKFTrackxgboostmodel.max_depth        =  {"min":1,"max":3  ,"value":3 }
#NewKFTrackxgboostmodel.gamma            =  {"min":0,"max":0.99,"value":	0.38212524090081945}
#NewKFTrackxgboostmodel.rate_drop        =  {"min":0,"max":1,"value":0.788588}
#NewKFTrackxgboostmodel.skip_drop        =  {"min":0,"max":1,"value":0.147907}
#NewKFTrackxgboostmodel.optimise()
#NewKFTrackxgboostmodel.train()
#NewKFTrackxgboostmodel.save_model("Models/NewKFTrack")
NewKFTrackxgboostmodel.load_model("Models/NewKFTrack")
NewKFTrackxgboostmodel.test()
#for i,item in enumerate(NewKFTrackxgboostmodel.y_predict_proba):
#    print(NewKFTrackxgboostmodel.DataSet.X_test.iloc[i].tolist())
#    print("prediction:",item)
NewKFTrackxgboostmodel.evaluate(plot=True,name="New KF Track Parameters")
#NewKFTrackxgboostmodel.plot_model()
NewKFTrackxgboostmodel.ONNX_convert_model("Models/NewKFTrack")
#NewKFTrackxgboostmodel.synth_model(hls=False,hdl=False,intwidth=24,fracwidth=16,plot=True,test_events=100000)

NewKFKFxgboostmodel = XGBoostClassifierModel()
NewKFKFxgboostmodel.load_data("NewKFKFDatasets/")
#NewKFKFxgboostmodel.comet_project_name = "Xgboost_newKF"
#NewKFKFxgboostmodel.min_child_weight =  {"min":0,"max":10, "value":7.554038941635881}
#NewKFKFxgboostmodel.alpha            =  {"min":0,"max":1,  "value":0.1455098913001549}
#NewKFKFxgboostmodel.early_stopping   =  {"min":1,"max":20, "value":5}
#NewKFKFxgboostmodel.learning_rate    =  {"min":0,"max":1,   "value":0.9513310026293323	}
#NewKFKFxgboostmodel.n_estimators     =  {"min":0,"max":100,   "value":74}
#NewKFKFxgboostmodel.subsample        =  {"min":0,"max":0.99,"value":0.9171935681586878	}
#NewKFKFxgboostmodel.max_depth        =  {"min":1,"max":3  ,"value":2 }
#NewKFKFxgboostmodel.gamma            =  {"min":0,"max":0.99,"value":	0.3653854859526517}
#NewKFKFxgboostmodel.rate_drop        =  {"min":0,"max":1,"value":0.788588}
#NewKFKFxgboostmodel.skip_drop        =  {"min":0,"max":1,"value":0.147907}
#NewKFKFxgboostmodel.optimise()
#NewKFKFxgboostmodel.train()#
#NewKFKFxgboostmodel.save_model("Models/NewKFKF")
NewKFKFxgboostmodel.load_model("Models/NewKFKF")
NewKFKFxgboostmodel.test()
#for i,item in enumerate(NewKFKFxgboostmodel.y_predict_proba):
#    print(NewKFKFxgboostmodel.DataSet.X_test.iloc[i].tolist())
 #   print("prediction:",item)

NewKFKFxgboostmodel.evaluate(plot=True,name="New KF KF Parameters")
NewKFKFxgboostmodel.plot_model()

NewKFKFxgboostmodel.ONNX_convert_model("Models/NewKFKF")
#NewKFTrackxgboostmodel.synth_model(hls=False,hdl=False,intwidth=24,fracwidth=16,plot=True,test_events=100000)


OldKFTrackxgboostmodel = XGBoostClassifierModel()
OldKFTrackxgboostmodel.load_data("OldKFTrackDatasets/")
#OldKFTrackxgboostmodel.comet_project_name = "Xgboost_oldKF"
#OldKFTrackxgboostmodel.optimise()
#OldKFTrackxgboostmodel.min_child_weight =  {"min":0,"max":10, "value":8.848520598251687	}
#OldKFTrackxgboostmodel.alpha            =  {"min":0,"max":1,  "value":0.2923963839055528 }
#OldKFTrackxgboostmodel.early_stopping   =  {"min":1,"max":20, "value":5}
#OldKFTrackxgboostmodel.learning_rate    =  {"min":0,"max":1,   "value":0.61217681482078}
#OldKFTrackxgboostmodel.n_estimators     =  {"min":0,"max":100,   "value":83}
#OldKFTrackxgboostmodel.subsample        =  {"min":0,"max":0.99,"value":0.38499426500459616 }
#OldKFTrackxgboostmodel.max_depth        =  {"min":1,"max":3  ,"value":2 }
#OldKFTrackxgboostmodel.gamma            =  {"min":0,"max":0.99,"value":	0.4848495035336481 }
#OldKFTrackxgboostmodel.rate_drop        =  {"min":0,"max":1,"value":0.788588}
#OldKFTrackxgboostmodel.skip_drop        =  {"min":0,"max":1,"value":0.147907}
#OldKFTrackxgboostmodel.train()#
OldKFTrackxgboostmodel.save_model("Models/OldKFTrack")
#OldKFTrackxgboostmodel.load_model("Models/OldKFTrack")
OldKFTrackxgboostmodel.test()
#for i,item in enumerate(OldKFTrackxgboostmodel.y_predict_proba):
#    print(OldKFTrackxgboostmodel.DataSet.X_test.iloc[i].tolist())
#    print("prediction:",item)
OldKFTrackxgboostmodel.evaluate(plot=True,name="Old KF Track Parameters")
#OldKFTrackxgboostmodel.ONNX_convert_model("Models/OldKFTrack")
#OldKFTrackxgboostmodel.plot_model()

OldKFChi2Model = CutClassifierModel()
OldKFChi2Model.load_data("OldKFFloatingTrackdatasets/")
OldKFChi2Model.test()
OldKFChi2Model.evaluate(plot=True,name="Old KF Track chi2")


NewKFChi2Model = CutClassifierModel()
NewKFChi2Model.load_data("NewKFFloatingTrackDatasets/")
NewKFChi2Model.test()
NewKFChi2Model.evaluate(plot=True,name="New KF Track chi2")
'''