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
    def evaluate(self,name="model",plot=False,binned=True):
        roc_auc = metrics.roc_auc_score(self.DataSet.y_test,self.y_predict_proba)
        binary_accuracy = metrics.accuracy_score(self.DataSet.y_test,self.y_predict)
        print(str(name) + " ROC AUC: ",roc_auc)
        print(str(name) + " ACC: ",binary_accuracy)
        if binned:
            roc_auc = metrics.roc_auc_score(self.DataSet.y_test,self.y_predict_binned)
            print(str(name) + " binned ROC AUC: ",roc_auc)
        if plot:
            self.calculate_roc(name,binned=binned)
        return(roc_auc,binary_accuracy)

    def calculate_roc(self,name,binned=True): 

        A = self.y_predict_proba
        B = self.DataSet.y_test.to_numpy()
        Q = self.y_predict_binned

        fpr, tpr, thresholds = metrics.roc_curve(B ,A, pos_label=1)
        auc = metrics.roc_auc_score(B,A)

        if binned:
            qfpr, qtpr, qthresholds = metrics.roc_curve(B ,Q, pos_label=1)
            qauc = metrics.roc_auc_score(B,Q)

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

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import mplhep as hep
        #hep.set_style("CMSTex")

        hep.cms.label()
        hep.cms.text("Simulation")
        plt.style.use(hep.style.CMS)

        SMALL_SIZE = 20
        MEDIUM_SIZE = 25
        BIGGER_SIZE = 30

        LEGEND_WIDTH = 20
        LINEWIDTH = 3
        MARKERSIZE = 20

        colormap = "jet"

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('axes', linewidth=LINEWIDTH)              # thickness of axes
        plt.rc('xtick', labelsize=SMALL_SIZE+2)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE-2)            # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        matplotlib.rcParams['xtick.major.size'] = 10
        matplotlib.rcParams['xtick.major.width'] = 3
        matplotlib.rcParams['xtick.minor.size'] = 10
        matplotlib.rcParams['xtick.minor.width'] = 3

        matplotlib.rcParams['ytick.major.size'] = 20
        matplotlib.rcParams['ytick.major.width'] = 3
        matplotlib.rcParams['ytick.minor.size'] = 10
        matplotlib.rcParams['ytick.minor.width'] = 2

        
        fig, ax = plt.subplots(1,2, figsize=(18,9)) 
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])


        #ax[0].set_title("ROC Curve",loc='left',fontsize=20)
        ax[0].plot(fpr,tpr,label=str(name)+ " AUC: %.3f"%auc,linewidth=2)
        ax[0].plot(qfpr,qtpr,label=str(name)+ "binned AUC: %.3f"%qauc,linewidth=2)
        ax[0].set_xlim([0,0.5])
        ax[0].set_ylim([0.5,1.0])
        ax[0].set_xlabel("Fake Positive Rate",ha="right",x=1)
        ax[0].set_ylabel("Identification Efficiency",ha="right",y=1)
        ax[0].legend()
        ax[0].grid()

        #ax[1].set_title("Normalised Class Distribution" ,loc='left',fontsize=20)
        ax[1].hist(genuine,color='g',bins=20,range=(0,1),histtype="step",label="Genuine",density=True,linewidth=2)
        ax[1].hist(fake,color='r',bins=20,range=(0,1),histtype="step",label="Fake",density=True,linewidth=2)



        self.output_bins

        ax[1].hist(qgenuine,color='darkgreen',bins=self.output_bins,histtype="step",label="Genuine binned",density=True,linewidth=2,linestyle='--')
        ax[1].hist(qfake,color='maroon',bins=self.output_bins,histtype="step",label="Fake binned",density=True,linewidth=2,linestyle='--')

        ax[1].grid()
        ax[1].set_yscale("log")
        ax[1].set_xlabel("BDT Output",ha="right",x=1)
        ax[1].set_ylabel("a.u.",ha="right",y=1)
        ax[1].legend(loc="upper center")

        plt.tight_layout()
        plt.savefig(str(name) + "ROC.png",dpi=600)
        plt.clf()

        fig, ax = plt.subplots(1,2, figsize=(18,9)) 
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])


        ax[0].set_title("TPR",loc='left',fontsize=20)
        ax[0].plot(thresholds,tpr,label=str(name),linewidth=2)
        ax[0].plot(qthresholds,qtpr,label="binned "+ str(name),linewidth=2)
        ax[0].set_xlim([0,1.0])
        ax[0].set_ylim([0.0,1.0])
        ax[0].set_xlabel("Threshold",ha="right",x=1)
        ax[0].set_ylabel("Identification Efficiency",ha="right",y=1)
        ax[0].legend()
        ax[0].grid()

        ax[1].set_title("FPR" ,loc='left',fontsize=20)
        ax[1].plot(thresholds,fpr,label=str(name),linewidth=2)
        ax[1].plot(qthresholds,qfpr,label="binned "+str(name),linewidth=2)
        ax[1].set_xlim([0,1.0])
        ax[1].set_ylim([0.0,1.0])
        ax[1].set_xlabel("Threshold",ha="right",x=1)
        ax[1].set_ylabel("False Positive Rate",ha="right",y=1)
        ax[1].legend()
        ax[1].grid()

        plt.tight_layout()
        plt.savefig(str(name) + "FPRTPR.png",dpi=600)
        plt.clf()
    
    def save_model(self,filepath):
        joblib.dump(self.model,filepath + ".pkl")
    def synth_model(self):
        pass
