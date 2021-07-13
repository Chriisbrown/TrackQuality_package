from Dataset import DataSet
from sklearn import metrics
import numpy as np
import joblib
from pathlib import Path
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mplhep as hep
#hep.set_style("CMSTex")
hep.cms.label()
hep.cms.text("Simulation")

plt.style.use(hep.cms.style.ROOT)

class TrackClassifierModel:
    def __init__(self):
        self.DataSet = None
        
        self.model = None
        self.y_predict = None
        self.y_predict_proba = None

        self.synth_y_predict = None
        self.synth_y_predict_proba = None

        self.comet_api_key = "expKifKow3Mn4dnjc1UGGOqrg"
        self.comet_project_name = "sklearn_test"
    
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
        pass
    def optimise(self):
        pass
    def evaluate(self,name="model",plot=False,quantised=True):
        roc_auc = metrics.roc_auc_score(self.DataSet.y_test,self.y_predict_proba)
        binary_accuracy = metrics.accuracy_score(self.DataSet.y_test,self.y_predict)
        print(str(name) + " ROC AUC: ",roc_auc)
        print(str(name) + " ACC: ",binary_accuracy)
        if quantised:
            roc_auc = metrics.roc_auc_score(self.DataSet.y_test,self.y_predict_quantised)
            print(str(name) + " Quantised ROC AUC: ",roc_auc)
        if plot:
            self.calculate_roc(name,quantised=quantised)
        return(roc_auc,binary_accuracy)

    def calculate_roc(self,name,quantised=True): 

        A = self.y_predict_proba
        B = self.DataSet.y_test.to_numpy()
        Q = self.y_predict_quantised

        fpr, tpr, thresholds = metrics.roc_curve(B ,A, pos_label=1)
        auc = metrics.roc_auc_score(B,A)

        if quantised:
            qfpr, qtpr, qthresholds = metrics.roc_curve(B ,Q, pos_label=1)
            qauc = metrics.roc_auc_score(B,Q)

        genuine = []
        fake = []

        for i in range(len(A)):
            if B[i] == 1:
                genuine.append(A[i])
            else:
                fake.append(A[i])

        
        fig, ax = plt.subplots(1,2, figsize=(18,9)) 
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)

        ax[0].set_title("ROC Curve",loc='left',fontsize=20)
        ax[0].plot(fpr,tpr,label=str(name)+ " AUC: %.3f"%auc)
        ax[0].plot(qfpr,qtpr,label=str(name)+ "Quantised AUC: %.3f"%qauc)
        ax[0].set_xlim([0,0.5])
        ax[0].set_ylim([0.5,1.0])
        ax[0].set_xlabel("Fake Positive Rate",ha="right",x=1,fontsize=16)
        ax[0].set_ylabel("Identification Efficiency",ha="right",y=1,fontsize=16)
        ax[0].legend()
        ax[0].grid()

        ax[1].set_title("Normalised Class Distribution" ,loc='left',fontsize=20)
        ax[1].hist(genuine,color='g',bins=20,range=(0,1),histtype="step",label="Genuine",density=True,linewidth=2)
        ax[1].hist(fake,color='r',bins=20,range=(0,1),histtype="step",label="Fake",density=True,linewidth=2)

        ax[1].hist(genuine,color='darkgreen',bins=8,range=(0,1),histtype="step",label="Genuine Quantised",density=True,linewidth=2,linestyle='--')
        ax[1].hist(fake,color='maroon',bins=8,range=(0,1),histtype="step",label="Fake Quantised",density=True,linewidth=2,linestyle='--')

        ax[1].grid()
        ax[1].set_yscale("log")
        ax[1].set_xlabel("BDT Output",ha="right",x=1,fontsize=16)
        ax[1].set_ylabel("a.u.",ha="right",y=1,fontsize=16)
        ax[1].legend(loc="upper center")

        plt.tight_layout()
        plt.savefig(str(name) + "ROC.png",dpi=600)
        plt.clf()

        fig, ax = plt.subplots(1,2, figsize=(18,9)) 
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)

        ax[0].set_title("TPR",loc='left',fontsize=20)
        ax[0].plot(thresholds,tpr,label=str(name))
        ax[0].plot(qthresholds,qtpr,label="Quantised "+ str(name))
        ax[0].set_xlim([0,1.0])
        ax[0].set_ylim([0.0,1.0])
        ax[0].set_xlabel("Threshold",ha="right",x=1,fontsize=16)
        ax[0].set_ylabel("Identification Efficiency",ha="right",y=1,fontsize=16)
        ax[0].legend()
        ax[0].grid()

        ax[1].set_title("FPR" ,loc='left',fontsize=20)
        ax[1].plot(thresholds,fpr,label=str(name))
        ax[1].plot(qthresholds,qfpr,label="Quantised "+str(name))
        ax[1].set_xlim([0,1.0])
        ax[1].set_ylim([0.0,1.0])
        ax[1].set_xlabel("Threshold",ha="right",x=1,fontsize=16)
        ax[1].set_ylabel("False Positive Rate",ha="right",y=1,fontsize=16)
        ax[1].legend()
        ax[1].grid()

        plt.tight_layout()
        plt.savefig(str(name) + "FPRTPR.png",dpi=600)
        plt.clf()
    
    def save_model(self,filepath):
        joblib.dump(self.model,filepath + ".pkl")
    def synth_model(self):
        pass