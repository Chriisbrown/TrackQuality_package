import uproot
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from plotting import *
from Utils.Formats import trackword_config
import gc
import time

with open("NoDegredation_Zp/Efficiencies/NoDegredationBDT_effs.pkl", 'rb') as f:
    BDT_effs_NoDegredation = pickle.load(f)
# with open("NoDegredation_Zp"+ "/Efficiencies/"+"Degredation9"+"BDT_effs.pkl", 'rb') as f:
#     BDT_effs_Degredation9 = pickle.load(f)
# with open("NoDegredation_Zp"+ "/Efficiencies/"+"Retrained"+"BDT_effs.pkl", 'rb') as f:
#     BDT_effs_Retrained = pickle.load(f)
# with open("NoDegredation_Zp"+ "/Efficiencies/"+"MVA1"+"BDT_effs.pkl", 'rb') as f:
#     BDT_effs_MVA1 = pickle.load(f)
with open("NoDegredation_Zp"+ "/Efficiencies/baseline_effs.pkl", 'rb') as f:
    baseline_effs = pickle.load(f)
with open("NoDegredation_Zp"+ "/Efficiencies/chi2_effs.pkl", 'rb') as f:
    chi2_effs = pickle.load(f)

# plot_eff_fake_curve([chi2_effs,BDT_effs_NoDegredation,BDT_effs_Degredation9,BDT_effs_Retrained],
#                     ['$\\chi^2$ Cuts','No Degredation BDT','Degredation 9 BDT','Continued Training BDT'],"NoDegredation")

with open("Degredation9_Zp"+ "/Efficiencies/"+"NoDegredation"+"BDT_effs.pkl", 'rb') as f:
    BDT_effs_NoDegredation_on9 = pickle.load(f)
# with open("Degredation9_Zp"+ "/Efficiencies/"+"Degredation9"+"BDT_effs.pkl", 'rb') as f:
#     BDT_effs_Degredation9 = pickle.load(f)
# with open("Degredation9_Zp"+ "/Efficiencies/"+"Retrained"+"BDT_effs.pkl", 'rb') as f:
#     BDT_effs_Retrained = pickle.load(f)
# with open("Degredation9_Zp"+ "/Efficiencies/"+"MVA1"+"BDT_effs.pkl", 'rb') as f:
#     BDT_effs_MVA1 = pickle.load(f)
with open("Degredation9_Zp"+ "/Efficiencies/baseline_effs.pkl", 'rb') as f:
    baseline_effs_on9 = pickle.load(f)
with open("Degredation9_Zp"+ "/Efficiencies/chi2_effs.pkl", 'rb') as f:
    chi2_effs = pickle.load(f)

# plot_eff_fake_curve([chi2_effs,BDT_effs_NoDegredation,BDT_effs_Degredation9,BDT_effs_Retrained],
#                     ['$\\chi^2$ Cuts','No Degredation BDT','Degredation 9 BDT','Continued Training BDT'],"Degredation9")

plot_eff_fake_curve([baseline_effs,baseline_effs_on9,BDT_effs_NoDegredation,BDT_effs_NoDegredation_on9],
                    ['Baseline Efficiency, No Degredation','Baseline Efficiency, 10\\% bad stubs','BDT Efficiency, No Degredation','BDT Efficiency, 10\\% bad stubs'],"NoDegredation_on9")




plot_dict = {"pt" :       {"xrange":[0,100],        "yrange":[0,1,0,0.2],  "bins":50,  "Latex":"$p_T$ [GeV]", "Loc":"Pt",   },
             "eta" :      {"xrange":[-2.4,2.4],     "yrange":[0,1,0,0.06], "bins":50,  "Latex":"$\\eta$",     "Loc":"Eta",  },
             "phi" :      {"xrange":[-np.pi,np.pi], "yrange":[0,1,0,0.06], "bins":450, "Latex":"$\\phi_0$",   "Loc":"Phi0", },
             "z0" :       {"xrange":[-20,20],       "yrange":[0,1,0,0.3],  "bins":50,  "Latex":"$z_0$ [cm]",  "Loc":"Z0",   }}
