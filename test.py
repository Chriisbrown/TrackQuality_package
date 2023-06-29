from GBDTTrackQualityModel import XGBoostClassifierModel,SklearnClassifierModel
from CutTrackQualityModel import CutClassifierModel
from Dataset import *
import numpy as np
import os

# AllKFDataset = KFDataSet("KFDataset_WithAll_9K",withstubs=True,withchi=True,withmatrix=True)
# AllKFDataset.load_data_from_root("/home/cb719/Documents/DataSets/TrackQuality/TrackNtuple",9000)
# AllKFDataset.generate_test_train()
# AllKFDataset.save_test_train_h5("AllKFDatasets9K/")

# Dataset = TrackDataSet("NewKF_TTTrackDataset_TT")
# Dataset.load_data_from_root("/home/cb719/Documents/Datasets/VertexDatasets/NewKFGTTData_New12/TrackNtuple_TT_new12.root",300874)#297676)
# Dataset.generate_test_train()
# Dataset.save_test_train_h5("NewKF_TTnew12/")


OldKFTrackxgboostmodel = XGBoostClassifierModel()
OldKFTrackxgboostmodel.load_data("NewKF_TTnew12/")
#OldKFTrackxgboostmodel.comet_project_name = "Xgboost_BDT_new12_newKF"
#OldKFTrackxgboostmodel.optimise()
#OldKFTrackxgboostmodel.min_child_weight =  {"min":0,"max":10, "value":	3.514259328631903	}
#OldKFTrackxgboostmodel.alpha            =  {"min":0,"max":1,  "value":0.012552137787942674	 }
#OldKFTrackxgboostmodel.early_stopping   =  {"min":1,"max":20, "value":5}
#OldKFTrackxgboostmodel.learning_rate    =  {"min":0,"max":1,   "value":0.11320393505755122}
#OldKFTrackxgboostmodel.n_estimators     =  {"min":0,"max":100,   "value":64}
#OldKFTrackxgboostmodel.subsample        =  {"min":0,"max":0.99,"value":0.05741151824193903 }
#OldKFTrackxgboostmodel.max_depth        =  {"min":1,"max":3  ,"value":2 }
#OldKFTrackxgboostmodel.gamma            =  {"min":0,"max":100,"value":	0.1 }
#OldKFTrackxgboostmodel.rate_drop        =  {"min":0,"max":1,"value":0.788588}
#OldKFTrackxgboostmodel.skip_drop        =  {"min":0,"max":1,"value":0.147907}
OldKFTrackxgboostmodel.train()#
OldKFTrackxgboostmodel.save_model("Models/Track_new12_NewKF")
OldKFTrackxgboostmodel.load_model("Models/Track_new12_NewKF")
#print(OldKFTrackxgboostmodel.model.predict_proba(np.expand_dims(np.array([3.85449,-0.146484,3,6,1,0,3]),axis=0)))

#OldKFTrackxgboostmodel.test()
#for i,item in enumerate(OldKFTrackxgboostmodel.y_predict_proba):
#    print(OldKFTrackxgboostmodel.DataSet.X_test.iloc[i].tolist())
#    print("prediction:",item)
#OldKFTrackxgboostmodel.evaluate(plot=True,name="BDT_new12_NewKF")
#OldKFTrackxgboostmodel.ONNX_convert_model("Models/Track_new12_NewKF")
#OldKFTrackxgboostmodel.plot_model()
OldKFTrackxgboostmodel.synth_model()

#OldKFChi2Model = CutClassifierModel()
#OldKFChi2Model.load_data("FullTT12/")
#OldKFChi2Model.test()
#OldKFChi2Model.evaluate(plot=True,name="chi2")


# NewKFChi2Model = CutClassifierModel()
# NewKFChi2Model.load_data("NewKFFloatingTrackDatasets/")
# NewKFChi2Model.test()
# NewKFChi2Model.evaluate(plot=True,name="New KF Track chi2")
# '''
