from Datasets.Dataset import DataSet
from sklearn import metrics
import numpy as np
import joblib
import pickle
from pathlib import Path
import os
from Utils.util import *
from Utils.Formats import *

class TrackClassifierModel:
    def __init__(self,name=""):
        self.name = name
        self.DataSet = None
        
        self.model = None
        self.y_predict = None
        self.y_predict_binned = None
        self.y_predict_proba = None

        self.synth_y_predict = None
        self.synth_y_predict_proba = None

        self.eta_roc_dict = []
        self.pt_roc_dict =  []
        self.phi_roc_dict = []
        self.z0_roc_dict =  []
        self.MVA_roc_dict =  []
        self.roc_dict =  []

        self.comet_api_key = "expKifKow3Mn4dnjc1UGGOqrg"
        self.comet_project_name = "sklearn_test"

        self.training_features = ['TanL', 'RescaledAbsZ', 'bit_bendchi2', 'trk_nstub', 'nlay_miss', 'bit_chi2rphi', 'bit_chi2rz']

    @classmethod
    def load_from_pkl(cls,model_filename):
        pklclass = cls()
        pklclass.model = joblib.load(model_filename+".pkl")
        return pklclass

    def load_data(self,filepath):
        self.DataSet = DataSet.fromTrainTest(filepath)
        print(self.DataSet)

    def train(self):
        pass

    def test(self):
        self.y_predict_binned = self.y_predict_proba

    def optimise(self):
        pass

    def evaluate(self,save_dir="/",plot=False,binned=True,full_parameter_rocs=False):
        os.system("mkdir " + save_dir )

        self.roc_dict = CalculateROC(self.DataSet.y_test,self.y_predict_proba, self.DataSet.X_test["weight"])
        self.qroc_dict = CalculateROC(self.DataSet.y_test,self.y_predict_binned, self.DataSet.X_test["weight"])

        if full_parameter_rocs: self.parameter_rocs()
        
        roc_auc = metrics.roc_auc_score(self.DataSet.y_test,self.y_predict_proba, sample_weight = self.DataSet.X_test["weight"] )
        binary_accuracy = metrics.accuracy_score(self.DataSet.y_test,self.y_predict, sample_weight = self.DataSet.X_test["weight"])

        print(str(self.name) + " ROC AUC: ",roc_auc)
        print(str(self.name) + " ACC: ",binary_accuracy)

        roc_auc = metrics.roc_auc_score(self.DataSet.y_test,self.y_predict_binned, sample_weight = self.DataSet.X_test["weight"])
        print(str(self.name) + " binned ROC AUC: ",roc_auc)

        if plot:
            self.plot_rocs_classes(self.name,save_dir)

    def plot_rocs_classes(self,name,save_dir,binned=True): 

        A = self.y_predict_proba
        B = self.DataSet.y_test.to_numpy()
        Q = self.y_predict_binned

        genuine = []
        fake = []
        qgenuine = []
        qfake = []

        for i in range(len(A)):
            if B[i] == 1:
                genuine.append(A[i])
                qgenuine.append(Q[i])
            else:
                fake.append(A[i])
                qfake.append(Q[i])

        fig, ax = plt.subplots(1,2, figsize=(18,9)) 
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])

        ax[0].plot(self.roc_dict["fpr_mean"], self.roc_dict["tpr_mean"],label=str(name)+ " AUC: %.3f $\\pm$ %.3f"%(self.roc_dict["roc_auc_mean"],self.roc_dict["roc_auc_err"]),linewidth=2,color='r')
        ax[0].fill_between(self.roc_dict["fpr_mean"],  self.roc_dict["tpr_mean"] , self.roc_dict["tpr_mean"] - self.roc_dict["tpr_err"],alpha=0.5,color='r')
        ax[0].fill_between(self.roc_dict["fpr_mean"],  self.roc_dict["tpr_mean"],  self.roc_dict["tpr_mean"] + self.roc_dict["tpr_err"],alpha=0.5,color='r')

        ax[0].plot(self.qroc_dict["fpr_mean"], self.qroc_dict["tpr_mean"],label=str(name)+ " binned AUC: %.3f $\\pm$ %.3f"%(self.qroc_dict["roc_auc_mean"],self.qroc_dict["roc_auc_err"]),linewidth=2,color='maroon')
        ax[0].fill_between(self.qroc_dict["fpr_mean"],  self.qroc_dict["tpr_mean"] , self.qroc_dict["tpr_mean"] - self.qroc_dict["tpr_err"],alpha=0.5,color='maroon')
        ax[0].fill_between(self.qroc_dict["fpr_mean"],  self.qroc_dict["tpr_mean"],  self.qroc_dict["tpr_mean"] + self.qroc_dict["tpr_err"],alpha=0.5,color='maroon')
        ax[0].set_xlim([0,0.5])
        ax[0].set_ylim([0.5,1.0])
        ax[0].set_xlabel("Fake Positive Rate",ha="right",x=1)
        ax[0].set_ylabel("Identification Efficiency",ha="right",y=1)
        ax[0].legend()
        ax[0].grid()

        ax[1].hist(genuine,color='g',bins=20,range=(0,1),histtype="step",label="Genuine",density=True,linewidth=2)
        ax[1].hist(fake,color='r',bins=20,range=(0,1),histtype="step",label="Fake",density=True,linewidth=2)

        ax[1].hist(qgenuine,color='darkgreen',bins=self.output_bins,histtype="step",label="Genuine binned",density=True,linewidth=2,linestyle='--')
        ax[1].hist(qfake,color='maroon',bins=self.output_bins,histtype="step",label="Fake binned",density=True,linewidth=2,linestyle='--')

        ax[1].grid()
        ax[1].set_yscale("log")
        ax[1].set_xlabel("BDT Output",ha="right",x=1)
        ax[1].set_ylabel("a.u.",ha="right",y=1)
        ax[1].legend(loc="upper center")

        plt.tight_layout()
        plt.savefig( str(save_dir) + str(name)+"ROC.png",dpi=600)
        plt.clf()

        fig, ax = plt.subplots(1,2, figsize=(18,9)) 
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])

        ax[0].plot(self.roc_dict["thresholds"], self.roc_dict["tpr_mean"],label=str(name),linewidth=2,color='r')
        ax[0].fill_between(self.roc_dict["thresholds"],  self.roc_dict["tpr_mean"] , self.roc_dict["tpr_mean"] - self.roc_dict["tpr_err"],alpha=0.5,color='r')
        ax[0].fill_between(self.roc_dict["thresholds"],  self.roc_dict["tpr_mean"],  self.roc_dict["tpr_mean"] + self.roc_dict["tpr_err"],alpha=0.5,color='r')

        ax[0].plot(self.roc_dict["thresholds"], self.qroc_dict["tpr_mean"],label="binned " + str(name),linewidth=2,color='maroon')
        ax[0].fill_between(self.roc_dict["thresholds"],  self.qroc_dict["tpr_mean"] , self.qroc_dict["tpr_mean"] - self.qroc_dict["tpr_err"],alpha=0.5,color='maroon')
        ax[0].fill_between(self.roc_dict["thresholds"],  self.qroc_dict["tpr_mean"],  self.qroc_dict["tpr_mean"] + self.qroc_dict["tpr_err"],alpha=0.5,color='maroon')

        ax[0].set_xlim([0,1.0])
        ax[0].set_ylim([0.0,1.0])
        ax[0].set_xlabel("Threshold",ha="right",x=1)
        ax[0].set_ylabel("Identification Efficiency",ha="right",y=1)
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(self.roc_dict["thresholds"], self.roc_dict["fpr_mean"],label=str(name),linewidth=2,color='r')
        ax[1].fill_between(self.roc_dict["thresholds"], self.roc_dict["fpr_mean"] , self.roc_dict["fpr_mean"] - self.roc_dict["fpr_err"],alpha=0.5,color='r')
        ax[1].fill_between(self.roc_dict["thresholds"], self.roc_dict["fpr_mean"] , self.roc_dict["fpr_mean"] + self.roc_dict["fpr_err"],alpha=0.5,color='r')

        ax[1].plot(self.qroc_dict["thresholds"], self.qroc_dict["fpr_mean"],label="binned " + str(name),linewidth=2,color='maroon')
        ax[1].fill_between(self.qroc_dict["thresholds"],  self.qroc_dict["fpr_mean"] , self.qroc_dict["tpr_mean"] - self.qroc_dict["fpr_err"],alpha=0.5,color='maroon')
        ax[1].fill_between(self.qroc_dict["thresholds"],  self.qroc_dict["fpr_mean"],  self.qroc_dict["tpr_mean"] + self.qroc_dict["fpr_err"],alpha=0.5,color='maroon')

        ax[1].set_xlim([0,1.0])
        ax[1].set_ylim([0.0,1.0])
        ax[1].set_xlabel("Threshold",ha="right",x=1)
        ax[1].set_ylabel("False Positive Rate",ha="right",y=1)
        ax[1].legend()
        ax[1].grid()

        plt.tight_layout()
        plt.savefig( str(save_dir) + str(name) +"FPRTPR.png",dpi=600)
        plt.clf()
    
    def parameter_rocs(self):
        self.eta_roc_dict = (calculate_ROC_bins(self.DataSet,self.y_predict_proba,
                            variable=Parameter_config["eta"]["branch"],
                            var_range=Parameter_config["eta"]["range"],
                            n_bins=Parameter_config["eta"]["bins"]))
        self.pt_roc_dict = (calculate_ROC_bins(self.DataSet,self.y_predict_proba,
                            variable=Parameter_config["pt"]["branch"],
                            var_range=Parameter_config["pt"]["range"],
                            n_bins=Parameter_config["pt"]["bins"]))
        self.phi_roc_dict = (calculate_ROC_bins(self.DataSet,self.y_predict_proba,
                            variable=Parameter_config["phi"]["branch"],
                            var_range=Parameter_config["phi"]["range"],
                            n_bins=Parameter_config["phi"]["bins"]))
        self.z0_roc_dict = (calculate_ROC_bins(self.DataSet,self.y_predict_proba,
                            variable=Parameter_config["z0"]["branch"],
                            var_range=Parameter_config["z0"]["range"],
                            n_bins=Parameter_config["z0"]["bins"]))

    def save_model(self,filepath,name):
        pass

    def load_model(self,filepath,name):
        pass

    def synth_model(self):
        pass

    def full_save(self,filepath,name):

        if not Path(filepath).is_dir():
            os.system("mkdir -p " + filepath)

        self.save_model(filepath,name)

        np.save(filepath+name+"y_predict.npy", self.y_predict)
        np.save(filepath+name+"y_predict_proba.npy", self.y_predict_proba)

        pickle.dump(self.eta_roc_dict, open(filepath + name + "eta_roc_dict" + ".pkl", "wb"))
        pickle.dump(self.pt_roc_dict, open(filepath + name + "pt_roc_dict" + ".pkl", "wb"))
        pickle.dump(self.phi_roc_dict, open(filepath + name + "phi_roc_dict" + ".pkl", "wb"))
        pickle.dump(self.z0_roc_dict, open(filepath + name + "z0_roc_dict" + ".pkl", "wb"))
        pickle.dump(self.MVA_roc_dict, open(filepath + name + "MVA_roc_dict" + ".pkl", "wb"))
        pickle.dump(self.roc_dict, open(filepath + name + "roc_dict" + ".pkl", "wb"))

    def full_load(self,filepath,name):

        self.load_model(filepath,name)

        self.y_predict = np.load(filepath+ name +"y_predict.npy")
        self.y_predict_proba = np.load(filepath+ name +"y_predict_proba.npy")

        self.eta_roc_dict = pickle.load(open(filepath + name + "eta_roc_dict" + ".pkl", "rb"))
        self.pt_roc_dict =  pickle.load(open(filepath + name + "pt_roc_dict" + ".pkl", "rb"))
        self.phi_roc_dict = pickle.load(open(filepath + name + "phi_roc_dict" + ".pkl", "rb"))
        self.z0_roc_dict =  pickle.load(open(filepath + name + "z0_roc_dict" + ".pkl", "rb"))
        self.MVA_roc_dict = pickle.load(open(filepath + name + "MVA_roc_dict" + ".pkl", "rb"))
        self.roc_dict =     pickle.load(open(filepath + name + "roc_dict" + ".pkl", "rb"))

