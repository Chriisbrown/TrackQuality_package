from GBDTTrackQualityModel import XGBoostClassifierModel,SklearnClassifierModel
from CutTrackQualityModel import CutClassifierModel
from Dataset import *
import numpy as np
import os

#newKFFloatingTrackDataset = FloatingTrackDataSet("NewKFTrack_Dataset_100K")
#newKFFloatingTrackDataset.load_data_from_root("/home/cb719/Documents/DataSets/NewKF_TTbar_170K_quality",100000)
#newKFFloatingTrackDataset.generate_test_train()
#newKFFloatingTrackDataset.save_test_train_h5("NewKFFloatingTrackDatasets/")

#newKFKFDataset = KFDataSet("NewKFKF_Dataset_100K")
#newKFKFDataset.load_data_from_root("/home/cb719/Documents/DataSets/NewKF_TTbar_170K_quality",100000)
#newKFKFDataset.generate_test_train()
#newKFKFDataset.save_test_train_h5("NewKFKFDatasets/")

#hybridFloatingDataset = FloatingTrackDataSet("OldKFTrack_Dataset_190K")
#hybridFloatingDataset.load_data_from_root("/home/cb719/Documents/DataSets/NewKF_TTbar_170K_quality",100000)
#hybridFloatingDataset.generate_test_train()
#hybridFloatingDataset.save_test_train_h5("OldKFFloatingTrackdatasets/")

#newKFTrackDataset = TrackDataSet("NewKFTrack_Dataset_100K")
#newKFTrackDataset.load_data_from_root("/home/cb719/Documents/DataSets/NewKF_TTbar_170K_quality",100000)
#newKFTrackDataset.generate_test_train()
#newKFTrackDataset.save_test_train_h5("NewKFTrackDatasets/")

#newKFKFDataset = KFDataSet("NewKFKF_Dataset_test")
#newKFKFDataset.load_data_from_root("/home/cb719/newKF",1)
#newKFKFDataset.generate_test()
#newKFKFDataset.save_test_train_h5("NewKFKFTest/")


#hybridDataset = TrackDataSet("OldKFTrack_Dataset_100K")
#hybridDataset.load_data_from_root("/home/cb719/Documents/DataSets/OldKF_TTbar_170K_quality",100000)
#hybridDataset.generate_test_train()
#hybridDataset.save_test_train_h5("OldKFTrackDatasets/")

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
'''
OldKFChi2Model = CutClassifierModel()
OldKFChi2Model.load_data("OldKFFloatingTrackdatasets/")
OldKFChi2Model.test()
OldKFChi2Model.evaluate(plot=True,name="Old KF Track chi2")


NewKFChi2Model = CutClassifierModel()
NewKFChi2Model.load_data("NewKFFloatingTrackDatasets/")
NewKFChi2Model.test()
NewKFChi2Model.evaluate(plot=True,name="New KF Track chi2")
