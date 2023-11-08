from Datasets.Dataset import *
from Utils.util import *
import numpy as np
import os
from Models.GBDTTrackQualityModel import XGBoostClassifierModel

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

save = False

folder_list = ["NoDegredation"]
[folder_list.append("Degredation"+str(i)) for i in range(1,10)] 

dataset_list = ["_"+str(i) for i in range(1,10)]
dataset_list.append("_Train")
dataset_list.append("_Zp")

if save:
    for folder in folder_list:
        for dataset in dataset_list:
            name = (folder + dataset)

            Dataset = TrackDataSet(name)
            Dataset.load_data_from_root("/home/cb719/Documents/Datasets/TrackDatasets/"+folder+"/"+name+"_TrackNtuple.root",300000)#297676)

            if dataset != "_Zp":
                Dataset.generate_train()
            else:
                Dataset.generate_test()

            Dataset.save_test_train_h5("Datasets/"+folder+"/"+name+"/")

for folder in folder_list:
    model = XGBoostClassifierModel(folder)
    model.load_model("Projects/"+folder+"/Models/",folder)

    eta_roc_dict = {x:[] for x in range(len(folder_list))}
    pt_roc_dict =  {x:[] for x in range(len(folder_list))}
    phi_roc_dict = {x:[] for x in range(len(folder_list))}
    z0_roc_dict =  {x:[] for x in range(len(folder_list))}

    fig, ax = plt.subplots(1,1, figsize=(10,10)) 
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

    os.system("mkdir -p " + "Projects/"+folder+"/Scan/" )
    #os.system("mkdir " + "Projects/NoDegredation/MCScan_MVA1/" )
    for i,folder in enumerate(folder_list):
        model.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
        #NoDegredation.load_data("Datasets/"+folder+"/")
        model.test()
        model.DataSet.X_test.reset_index(inplace=True)

        predictions = model.y_predict_proba #NoDegredation.DataSet.X_test["trk_MVA1"]  #NoDegredation.y_predict_proba

        fpr,tpr,fpr_err,tpr_err,_,__,auc,auc_err = CalculateROC(model.DataSet.y_test,predictions)
        eta_roc_dict[i] = (calculate_ROC_bins(model.DataSet,predictions,variable="trk_eta",var_range=[-2.4,2.4],n_bins=20))
        pt_roc_dict[i] = (calculate_ROC_bins(model.DataSet,predictions,variable="trk_pt",var_range=[2,100],n_bins=10))
        phi_roc_dict[i] = (calculate_ROC_bins(model.DataSet,predictions,variable="trk_phi",var_range=[-3.14,3.14],n_bins=20))
        z0_roc_dict[i] = (calculate_ROC_bins(model.DataSet,predictions,variable="trk_z0",var_range=[-15,15],n_bins=20))

        ax.plot(fpr, tpr,label=folder+ " AUC: %.3f $\\pm$ %.3f"%(auc,auc_err),linewidth=2,color=colours[i])
        ax.fill_between(fpr,  tpr , tpr - tpr_err,alpha=0.5,color=colours[0])
        ax.fill_between(fpr,  tpr,  tpr + tpr_err,alpha=0.5,color=colours[0])


    ax.set_xlim([0.0,0.4])
    ax.set_ylim([0.8,1.0])
    ax.set_xlabel("False Positive Rate",ha="right",x=1)
    ax.set_ylabel("Identification Efficiency",ha="right",y=1)
    ax.legend()
    ax.grid()
    plt.savefig("Projects/"+folder+"/Scan/ROC.png",dpi=600)
    plt.savefig("Projects/"+folder+"/Scan/ROC.pdf")
    plt.clf()

    plot_ROC_bins(eta_roc_dict,folder_list,"Projects/"+folder+"/Scan/",variable="trk_eta",var_range=[-2.4,2.4],n_bins=20,typesetvar="Track $\\eta$")
    plot_ROC_bins(pt_roc_dict,folder_list,"Projects/"+folder+"/Scan/",variable="trk_pt",var_range=[2,100],n_bins=10,typesetvar="Track $p_T$")
    plot_ROC_bins(phi_roc_dict,folder_list,"Projects/"+folder+"/Scan/",variable="trk_phi",var_range=[-3.14,3.14],n_bins=20,typesetvar="Track $\\phi$")
    plot_ROC_bins(z0_roc_dict,folder_list,"Projects/"+folder+"/Scan/",variable="trk_z0",var_range=[-15,15],n_bins=20,typesetvar="Track $z_0$")
