from TrackQualityModel import XGBoostClassifierModel
from Dataset import DataSet,EventDataSet
import numpy as np


TestDataset = EventDataSet("PV_170K")
TestDataset.load_data_from_root("/home/cb719/Documents/Datasets/TTbar_170K_hybrid",170000)

TestDataset.find_PV()
TestDataset.find_fast_hist()


for i in range(10):
    print(TestDataset.pv_list[i],TestDataset.z0List[i])

print (np.percentile(np.array(TestDataset.z0List)-np.array(TestDataset.pv_list),[5,15,50,85,95]))

#QualityOnlyDataset = DataSet("TrackQualityOnly300")
#QualityOnlyDataset.training_features = ["bit_bendchi2","bit_chi2rphi","bit_chi2rz","pred_nstub","bit_InvR","bit_TanL","bit_z0"]
#                                            0                1            2           3            4           5          6
#QualityOnlyDataset.load_data_from_root("/home/cb719/Documents/Datasets/TT_object_300k",300000)
#QualityOnlyDataset.generate_test_train()
#QualityOnlyDataset.save_test_train_h5("/home/cb719/Documents/Datasets/TT_training_qOnly_300k/")


#xgboostmodel = XGBoostClassifierModel() 
#xgboostmodel.load_data("/home/cb719/Documents/Datasets/TT_training_qOnly_300k/")
#xgboostmodel.comet_project_name = "Xgboost_quality_features"
#xgboostmodel.optimise()
#xgboostmodel.train()
#xgboostmodel.test()
#xgboostmodel.evaluate()
#xgboostmodel.plot_model()
#xgboostmodel.save_model("xgboost_model")
#xgboostmodel.synth_model()