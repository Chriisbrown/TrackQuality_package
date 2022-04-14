from GBDTTrackQualityModel import XGBoostClassifierModel,SklearnClassifierModel
from CutTrackQualityModel import CutClassifierModel
from Dataset import *
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")
hep.cms.label()
hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 30
MEDIUM_SIZE = 30
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=5)              # thickness of axes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=25)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 20
matplotlib.rcParams['xtick.major.width'] = 5
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 4

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 5
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 4
plt.clf()
#newKFFloatingTrackDataset = FloatingTrackDataSet("NewKFTrack_Dataset_100K")
newKFFloatingTrackDataset = FloatingTrackDataSet.fromTrainTest("../../NewKFFloatingTrackDatasets/")
fakes = newKFFloatingTrackDataset.y_train == 0
true = newKFFloatingTrackDataset.y_train == 1
plt.hist(newKFFloatingTrackDataset.X_train["trk_chi2rphi"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(500.0), 50))
plt.hist(newKFFloatingTrackDataset.X_train["trk_chi2rphi"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(500.0), 50))
plt.xscale("log")
plt.xlim(1,500)
plt.savefig("Chi2rphinewKF.png")
plt.clf()

plt.hist(newKFFloatingTrackDataset.X_train["trk_chi2rz"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(100.0), 50))
plt.hist(newKFFloatingTrackDataset.X_train["trk_chi2rz"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(100.0), 50))
plt.xscale("log")
plt.savefig("Chi2rznewKF.png")
plt.clf()
plt.hist(newKFFloatingTrackDataset.X_train["trk_bendchi2"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(30.0), 50))
plt.hist(newKFFloatingTrackDataset.X_train["trk_bendchi2"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(30.0), 50))
plt.xscale("log")

plt.savefig("Chi2bendnewKF.png")
plt.clf()

fig, ax = plt.subplots(1,1, figsize=(20,10)) 

ax.hist(newKFFloatingTrackDataset.X_train["trk_eta"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,range=(-2.4,2.4), bins= 50,label="Fakes",linewidth=3)
ax.hist(newKFFloatingTrackDataset.X_train["trk_eta"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,range=(-2.4,2.4), bins= 50, label="True",linewidth=3)
ax.set_yscale("log")
ax.legend(loc=8)
ax.grid()
ax.set_xlabel("Track $\\eta$",ha="right",x=1)
ax.set_ylabel("a.u.",ha="right",y=1)
plt.tight_layout()
plt.savefig("EtanewKF.png")

plt.clf()

plt.hist(newKFFloatingTrackDataset.X_train["trk_pt"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,range=(0,50), bins= 50)
plt.hist(newKFFloatingTrackDataset.X_train["trk_pt"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,range=(0,50), bins= 50)
plt.yscale("log")
plt.savefig("ptnewKF.png")
plt.clf()




#OldKFFloatingTrackDataset = FloatingTrackDataSet("OldKFTrack_Dataset_190K")
#OldKFFloatingTrackDataset.load_data_from_root("/home/cb719/Documents/DataSets/OldKF_TTbar_170K_quality",100000)
#OldKFFloatingTrackDataset.generate_test_train()
#OldKFFloatingTrackDataset.save_test_train_h5("../../OldKFFloatingTrackdatasets/")

OldKFFloatingTrackDataset = FloatingTrackDataSet.fromTrainTest("../../OldKFFloatingTrackdatasets/")
fakes = OldKFFloatingTrackDataset.y_train == 0
true = OldKFFloatingTrackDataset.y_train == 1
plt.hist(OldKFFloatingTrackDataset.X_train["trk_chi2rphi"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(500.0), 50))
plt.hist(OldKFFloatingTrackDataset.X_train["trk_chi2rphi"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(500.0), 50))
plt.xscale("log")
plt.xlim(1,500)
plt.savefig("Chi2rphiOldKF.png")
plt.clf()
plt.hist(OldKFFloatingTrackDataset.X_train["trk_chi2rz"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(100.0), 50))
plt.hist(OldKFFloatingTrackDataset.X_train["trk_chi2rz"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(100.0), 50))
plt.xscale("log")
plt.savefig("Chi2rzOldKF.png")
plt.clf()
plt.hist(OldKFFloatingTrackDataset.X_train["trk_bendchi2"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(30.0), 50))
plt.hist(OldKFFloatingTrackDataset.X_train["trk_bendchi2"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,bins=np.logspace(np.log10(0.1),np.log10(30.0), 50))
plt.xscale("log")
plt.savefig("Chi2bendOldKF.png")
plt.clf()
fig, ax = plt.subplots(1,1, figsize=(20,10)) 

ax.hist(OldKFFloatingTrackDataset.X_train["trk_eta"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,range=(-2.4,2.4), bins= 50,label="Fakes",linewidth=3)
ax.hist(OldKFFloatingTrackDataset.X_train["trk_eta"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,range=(-2.4,2.4), bins= 50, label="True",linewidth=3)
ax.set_yscale("log")
ax.legend(loc=8)
ax.grid()
ax.set_xlabel("Track $\\eta$",ha="right",x=1)
ax.set_ylabel("a.u.",ha="right",y=1)
plt.tight_layout()
plt.savefig("EtaOldKF.png")

plt.clf()

plt.hist(OldKFFloatingTrackDataset.X_train["trk_pt"][fakes.to_numpy()[:,0]],color='r',histtype="step",density=True,range=(0,50), bins= 50)
plt.hist(OldKFFloatingTrackDataset.X_train["trk_pt"][true.to_numpy()[:,0]],color='g',histtype="step",density=True,range=(0,50), bins= 50)
plt.yscale("log")
plt.savefig("ptoldKF.png")
plt.clf()





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