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
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-2)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 10
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 3

matplotlib.rcParams['ytick.major.size'] = 10
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 3

colours=["red","black","blue","orange","purple","goldenrod",'green',"yellow","turquoise","magenta"]
linestyles = ["-","--","dotted",(0, (3, 5, 1, 5)),(0, (3, 5, 1, 5, 1, 5)),(0, (3, 10, 1, 10)),(0, (3, 10, 1, 10, 1, 10))]


folder_list = ["NoDegredation"]
[folder_list.append("Degredation"+str(i)) for i in range(1,10)] 


os.system("mkdir " + "Projects/Scan/" )

sample_array = np.zeros([10,10])

for j,model_folder in enumerate(folder_list):
    print("#========================#")
    print("||                      ||")
    print("|| Model ",model_folder,"   ||")
    print("||                      ||")
    print("#========================#")
    model = XGBoostClassifierModel()
    model.load_model("Projects/"+model_folder+"/Models/",model_folder)

    for i,folder in enumerate(folder_list):
        model.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
        model.test()

        predictions =  model.y_predict_proba #model.DataSet.X_test["trk_MVA1"]
        fpr,tpr,fpr_err,tpr_err,_,__,auc,auc_err = CalculateROC(model.DataSet.y_test,predictions)

        sample_array[j][i] = auc

fig, ax = plt.subplots(1,1, figsize=(10,10)) 
hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)



im = ax.imshow(sample_array.T,cmap='viridis')
labels = [i for i in range(10)]
ax.set_ylabel("Sample", horizontalalignment='right', y=1.0)
ax.set_yticks(np.arange(len(labels))+0.5)
ax.set_yticklabels(labels)
ax.set_xlabel("Model ", horizontalalignment='right', x=1.0)
ax.set_xticks(np.arange(len(labels))+0.5)
ax.set_xticklabels(labels)

for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)

dx = -25/72; dy = 0/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

            # apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)


for tick in ax.yaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)

dx = 0/72; dy = 25/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

for label in ax.yaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

cbar = plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
cbar.set_label("ROC Score",labelpad=20)
plt.tight_layout()

plt.savefig("Projects/Scan/ROCscan.png",dpi=600)
plt.clf()
