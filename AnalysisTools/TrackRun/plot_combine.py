import uproot
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from plotting import *
import gc

outputFolder = "Combined/Plots/"

os.system("mkdir Combined")
os.system("mkdir Combined/Plots")

# dataset_list = ["TTv2","TTv5","TTv4","TTv1","TTv3"]
# dataset_names = ["No Degradation","BRI","BRI + 1% BS","BRI + 5% BS","BRI + 10% BS"]

# dataset_list = ["Zp","Zp_d1","Zp_d2", "Zp_d3",]
# dataset_names = ["No Degradation","L5 + 5%","L1 + 5%","L1 + L2"]

#dataset_list = [        "Zp",     "Zp_d1",   "Zp_d2", "Zp_d3",  "Zp_d4",  "Zp_d5","Zp_d6","Zp_d7",  "Zp_d8",  "Zp_d9"]
#dataset_names = ["No Degradation","L5 + 5%","L1 + 5%","L1 + L2","L1 + D1","5%",   "1%",   "L5 + 1%","L1 + 1%","10%"]

dataset_list = ["Zp",             "Zp_d3",  "Zp_d4",  "Zp_d8",  "Zp_d7",   "Zp_d2",  "Zp_d1" ,"Zp_d6","Zp_d5","Zp_d9"]
dataset_names = ["No Degradation","L1 + L2","L1 + D1","L1 + 1%","L5 + 1%","L1 + 5%","L5 + 5%","1%",   "5%",   "10%"]

# dataset_list = ["Large_TT_v1","Large_TT_v2","Large_TT_v3","Large_TT_v4","Large_TT_v5"]
# dataset_names = ["No Degradation","BRI","BRI + 1% BS","BRI + 5% BS","BRI + 10% BS"]

dataset_dataframes = []


for dataset in dataset_list:
    dataset_dataframes.append(pd.read_pickle(dataset+"/Arrays/track_data_frame.pkl"))

plt.close()
plt.clf()
figure=plot_multiple_variable([dataset["trk_eta"] for dataset in dataset_dataframes],
                        "Track $\\eta$",dataset_names,"",
                        xrange=(-2.4,2.4),yrange=None,density=False)
plt.savefig("%s/EtaCombinedDistribution.png" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable([dataset["trk_pt"] for dataset in dataset_dataframes]
                     ,"Track $p_T$ [GeV]",dataset_names,"",
                     xrange=(0,128),yrange=None,density=False)
plt.savefig("%s/PtCombinedDistribution.png" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_pt"] for dataset in dataset_dataframes]
                     ,"Track $p_T$ [GeV]",dataset_names,"",
                     xrange=(2,50),yrange=None,density=False,lloc="upper right",textpos=(30,340000))
plt.savefig("%s/PtCombinedDistributionRatio.png" % outputFolder,dpi=200)
plt.savefig("%s/PtCombinedDistributionRatio.pdf" % outputFolder,dpi=200)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_pt"] for dataset in dataset_dataframes]
                     ,"Track $p_T$ [GeV]",dataset_names,"",
                     xrange=(2,50),yrange=None,density=False,lloc="upper right",textpos=(30,340000),log=True)
plt.savefig("%s/PtCombinedDistributionRatioLog.png" % outputFolder)
plt.savefig("%s/PtCombinedDistributionRatioLog.pdf" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_eta"] for dataset in dataset_dataframes],
                        "Track $\\eta$",dataset_names,"",
                        xrange=(-2.4,2.4),yrange=None,density=False,lloc="lower center",textpos=(-1,7000))
plt.savefig("%s/EtaCombinedDistributionRatio.png" % outputFolder,dpi=200)
plt.savefig("%s/EtaCombinedDistributionRatio.pdf" % outputFolder,dpi=200)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_MVA1"] for dataset in dataset_dataframes],
                        "Track Quality BDT Output",dataset_names,"",
                        xrange=(0,1),yrange=None,density=False,log=True)
plt.savefig("%s/MVACombinedDistributionRatio.png" % outputFolder)
plt.savefig("%s/MVACombinedDistributionRatio.pdf" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_bendchi2"] for dataset in dataset_dataframes],
                        "Track $\\chi^2_{bend}$",dataset_names,"",
                        xrange=(0,15),yrange=None,density=False,lloc="upper right")
plt.savefig("%s/BendChi2CombinedDistributionRatio.png" % outputFolder)
plt.savefig("%s/BendChi2CombinedDistributionRatio.pdf" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_chi2rphi"] for dataset in dataset_dataframes],
                        "Track $\\chi^2_{r\\phi}$",dataset_names,"",
                        xrange=(0,15),yrange=None,density=False,lloc="upper right")
plt.savefig("%s/Chi2RphiCombinedDistributionRatio.png" % outputFolder)
plt.savefig("%s/Chi2RphiCombinedDistributionRatio.pdf" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_chi2rz"] for dataset in dataset_dataframes],
                        "Track $\\chi^2_{rz}$",dataset_names,"",
                        xrange=(0,15),yrange=None,density=False,lloc="upper right")
plt.savefig("%s/Chi2RzCombinedDistributionRatio.png" % outputFolder)
plt.savefig("%s/Chi2RzCombinedDistributionRatio.pdf" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_nstub"] for dataset in dataset_dataframes],
                        "Track # Stubs",dataset_names,"",
                        xrange=(3,7),yrange=None,density=False,bins=4,xratios=[4.5,5.5,6.5,7.5])
plt.savefig("%s/NstubCombinedDistributionRatio.png" % outputFolder)
plt.savefig("%s/NstubCombinedDistributionRatio.pdf" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_phi"] for dataset in dataset_dataframes],
                        "Track $\\phi$",dataset_names,"",
                        xrange=(-3.14,3.14),yrange=(0.8,1.2),density=False,lloc="lower center")
plt.savefig("%s/PhiCombinedDistributionRatio.png" % outputFolder)
plt.savefig("%s/PhiCombinedDistributionRatio.pdf" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["trk_z0"] for dataset in dataset_dataframes],
                        "Track $z_0$",dataset_names,"",
                        xrange=(-20,20),yrange=(0.5,1.2),density=False)
plt.savefig("%s/Z0CombinedDistributionRatio.png" % outputFolder)
plt.savefig("%s/Z0CombinedDistributionRatio.pdf" % outputFolder)

plt.close()
plt.clf()
figure=plot_multiple_variable_ratio([dataset["nlay_miss"] for dataset in dataset_dataframes],
                        "Track # Missing Internal Stubs",dataset_names,"",
                        xrange=(-1,3),yrange=(0.5,3.5),density=False,bins=4,xratios=[0.5,1.5,2.5,3.5],lloc="upper right")
plt.savefig("%s/NlaymissCombinedDistributionRatio.png" % outputFolder)
plt.savefig("%s/NlaymissCombinedDistributionRatio.pdf" % outputFolder)
