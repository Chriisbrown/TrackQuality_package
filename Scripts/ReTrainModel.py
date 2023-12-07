from Models.GBDTTrackQualityModel import XGBoostClassifierModel,TFDFClassifierModel
from Models.CutTrackQualityModel import CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os

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

colours=["red","black","blue","orange","purple","goldenrod",'green',"yellow","turquoise","magenta"]
linestyles = ["-","--","dotted",(0, (3, 5, 1, 5)),(0, (3, 5, 1, 5, 1, 5)),(0, (3, 10, 1, 10)),(0, (3, 10, 1, 10, 1, 10))]

OriginalModel = "NewKF_NoDegredation"
OriginalFolder = "NewKF_NoDegredation"

SecondModel = "NewKF_Degredation9"
SecondFolder = "NewKF_Degredation9"

RetrainModel = "NewKF_Retrained"
RetrainFolder = "NewKF_Retrained"

O_model = XGBoostClassifierModel("No Degredation")
O_model.load_model("Projects/"+OriginalFolder+"/Models/",OriginalModel+"_XGB_")

R_model = XGBoostClassifierModel("No Degredation Retrained on 9")
R_model.load_model("Projects/"+OriginalFolder+"/Models/",OriginalModel+"_XGB_")

S_model = XGBoostClassifierModel("Degredation 9")
S_model.load_model("Projects/"+SecondModel+"/Models/",SecondModel+"_XGB_")

R_model.load_data("Datasets/"+SecondFolder+"/"+SecondFolder+"_1/")
R_model.learning_rate = 0.1
R_model.n_estimators = 40
R_model.retrain()
R_model.save_model("Projects/"+RetrainFolder+"/Models/",RetrainModel+"_XGB_")
R_model.load_model("Projects/"+RetrainFolder+"/Models/",RetrainModel+"_XGB_")

C_model = CutClassifierModel("Cut Model")

models = [O_model,R_model,S_model]

folders = [OriginalFolder,SecondFolder]

for folder in folders:
    os.system("mkdir -p Projects/"+RetrainFolder+"/"+folder+"/Scan/")

    eta_roc_dict = {x:[] for x in range(len(models))}
    pt_roc_dict =  {x:[] for x in range(len(models))}
    phi_roc_dict = {x:[] for x in range(len(models))}
    z0_roc_dict =  {x:[] for x in range(len(models))}

    fig, ax = plt.subplots(1,1, figsize=(10,10)) 
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

    for i,model in enumerate(models):
        model.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
        model.test()

        roc_dict = CalculateROC(model.DataSet.y_test,model.y_predict_proba)

        eta_roc_dict[i] = (calculate_ROC_bins(model.DataSet,model.y_predict_proba,variable="trk_eta",var_range=[-2.4,2.4],n_bins=20))
        pt_roc_dict[i] = (calculate_ROC_bins(model.DataSet,model.y_predict_proba,variable="trk_pt",var_range=[2,100],n_bins=10))
        phi_roc_dict[i] = (calculate_ROC_bins(model.DataSet,model.y_predict_proba,variable="trk_phi",var_range=[-3.14,3.14],n_bins=20))
        z0_roc_dict[i] = (calculate_ROC_bins(model.DataSet,model.y_predict_proba,variable="trk_z0",var_range=[-15,15],n_bins=20))

        ax.plot(roc_dict['fpr_mean'], roc_dict['tpr_mean'],label=model.name+ " AUC: %.3f $\\pm$ %.3f"%(roc_dict['roc_auc_mean'],roc_dict['roc_auc_err']),linewidth=2,color=colours[i])
        ax.fill_between(roc_dict['fpr_mean'],  roc_dict['tpr_mean'] , roc_dict['tpr_mean'] - roc_dict['tpr_err'],alpha=0.5,color=colours[i])
        ax.fill_between(roc_dict['fpr_mean'],  roc_dict['tpr_mean'],  roc_dict['tpr_mean'] + roc_dict['tpr_err'],alpha=0.5,color=colours[i])

    ax.set_xlim([0.0,0.4])
    ax.set_ylim([0.8,1.0])
    ax.set_xlabel("False Positive Rate",ha="right",x=1)
    ax.set_ylabel("Identification Efficiency",ha="right",y=1)
    ax.legend()
    ax.grid()
    plt.savefig("Projects/"+RetrainFolder+"/"+folder+"/Scan/ROC.png",dpi=600)
    plt.clf()

    plot_ROC_bins(eta_roc_dict,[model.name for model in models],"Projects/"+RetrainFolder+"/"+folder+"/Scan/",variable="trk_eta",var_range=[-2.4,2.4],n_bins=20,typesetvar="Track $\\eta$")
    plot_ROC_bins(pt_roc_dict,[model.name for model in models],"Projects/"+RetrainFolder+"/"+folder+"/Scan/",variable="trk_pt",var_range=[2,100],n_bins=10,typesetvar="Track $p_T$")
    plot_ROC_bins(phi_roc_dict,[model.name for model in models],"Projects/"+RetrainFolder+"/"+folder+"/Scan/",variable="trk_phi",var_range=[-3.14,3.14],n_bins=20,typesetvar="Track $\\phi$")
    plot_ROC_bins(z0_roc_dict,[model.name for model in models],"Projects/"+RetrainFolder+"/"+folder+"/Scan/",variable="trk_z0",var_range=[-15,15],n_bins=20,typesetvar="Track $z_0$")

