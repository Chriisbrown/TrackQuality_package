import uproot
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from plotting import *
import gc

"python track_runner.py MCDegredation/LargeSamples/MC_Degredation1_TT_TrackNtuple.root Large_TT_v1 True 29"
"python track_runner.py RootFilename.root name_of_dataset save_arrays number_of_batches"
rootdir = "/home/cb719/Documents/DataSets/VertexDatasets/OldKFGTTData_New12/"
#rootdir = "/home/cb719/Documents/Datasets/TrackDatasets/"
f = sys.argv[1]
name = sys.argv[2]
save = sys.argv[3]

outputFolder = name + "/Plots/"

os.system("mkdir "+name)
os.system("mkdir "+name+ "/Plots")
os.system("mkdir "+name+ "/Plots/Trk")
os.system("mkdir "+name+ "/Plots/twoD")
os.system("mkdir "+name+ "/Plots/TP")
os.system("mkdir "+name+ "/Plots/TP/PUSplit")
os.system("mkdir "+name+ "/Plots/Trk/PUSplit")
os.system("mkdir "+name+ "/Plots/Trk/PDGIDSplit")
os.system("mkdir "+name+ "/Arrays")

track_data_frame = pd.DataFrame()
tp_data_frame = pd.DataFrame()
match_data_frame = pd.DataFrame()

max_batch = int(sys.argv[4])

track_feature_list = ['trk_pt', 'trk_eta', 'trk_phi', 
                      'trk_z0', 'trk_chi2', 'trk_chi2rphi', 
                      'trk_chi2rz', 'trk_bendchi2', 'trk_MVA1',
                      'trk_nstub', 'trk_fake', 'trk_matchtp_pdgid', 'trk_matchtp_pt',
                      'trk_matchtp_eta', 'trk_matchtp_phi', 'trk_matchtp_z0',
                      'trk_hitpattern','trk_nPSstub_hitpattern','trk_n2Sstub_hitpattern',
                      'trk_nLostPSstub_hitpattern','trk_nLost2Sstub_hitpattern',
                      'trk_nLoststub_V1_hitpattern','trk_nLoststub_V2_hitpattern'
                      ]

match_feature_list = ["matchtrk_MVA1"]

tp_feature_list = ['tp_pt','tp_eta', 'tp_phi', 'tp_dxy', 'tp_d0', 'tp_z0', 'tp_d0_prod',
                    'tp_z0_prod', 'tp_pdgid', 'tp_nstub', 'tp_eventid', 'tp_charge']

if save == "True":
    tracks = uproot.open(rootdir+f+":L1TrackNtuple/eventTree", num_workers=8)

    track_data_frame = save_branches(tracks,track_data_frame,track_feature_list,max_batch=max_batch,name="tracks")

    track_data_frame["nlay_miss"] = track_data_frame["trk_hitpattern"].apply(set_nlaymissinterior)

    track_data_frame["chi2rphi_dof"] = track_data_frame["trk_chi2rphi"]/(track_data_frame["trk_nstub"] - 2)
    track_data_frame["chi2rz_dof"] = track_data_frame["trk_chi2rz"]/(track_data_frame["trk_nstub"] - 2)

    track_data_frame["trk_TanL"] = track_data_frame["trk_eta"].apply(tanl)

    track_data_frame.to_pickle(name+"/Arrays/track_data_frame.pkl") 

    tp_data_frame = save_branches(tracks,tp_data_frame,tp_feature_list,max_batch=max_batch,name="tps")
    #match_data_frame = save_branches(tracks,match_data_frame,match_feature_list,max_batch=max_batch,name="match tracks")

    # Save, can modify these locations as needed

    
    tp_data_frame.to_pickle(name+"/Arrays/tp_data_frame.pkl") 
    match_data_frame.to_pickle(name+"/Arrays/match_data_frame.pkl") 

if save == "False":

    track_data_frame = pd.read_pickle(name+"/Arrays/track_data_frame.pkl") 
    tp_data_frame = pd.read_pickle(name+"/Arrays/tp_data_frame.pkl") 
    match_data_frame = pd.read_pickle(name+"/Arrays/match_data_frame.pkl") 

print(track_data_frame.head())
print(tp_data_frame.head())
print(match_data_frame.head())

plot_dict = {"trk_pt" :       {"dataframe":track_data_frame, "xrange":(0,128),        "density":True, "log":True,  "lloc":"upper right",  "bins":50,  "Latex":"Track $p_T$ [GeV]",              "SubFolder":"Trk/", "Loc":"Pt",       "PUsplit":True,  "PDGIDSplit":True,  "match":"trk_matchtp_pt",      },
             "trk_eta" :      {"dataframe":track_data_frame, "xrange":(-2.4,2.4),     "density":True, "log":False, "lloc":"upper center", "bins":50,  "Latex":"Track $\\eta$",                   "SubFolder":"Trk/", "Loc":"Eta",      "PUsplit":True,  "PDGIDSplit":True,  "match":"trk_matchtp_eta",     },
             "trk_phi" :      {"dataframe":track_data_frame, "xrange":(-np.pi,np.pi), "density":True, "log":False, "lloc":"lower center", "bins":450, "Latex":"Track $\\phi_0$",                 "SubFolder":"Trk/", "Loc":"Phi0",     "PUsplit":True,  "PDGIDSplit":True,  "match":"trk_matchtp_phi",     },
             "trk_z0" :       {"dataframe":track_data_frame, "xrange":(-15,15),       "density":True, "log":False, "lloc":"lower center", "bins":50,  "Latex":"Track $z_0$",                    "SubFolder":"Trk/", "Loc":"Z0",       "PUsplit":True,  "PDGIDSplit":True,  "match":"trk_matchtp_z0",      },
             "chi2rphi_dof" : {"dataframe":track_data_frame, "xrange":(0,40),         "density":True, "log":True,  "lloc":"upper right",  "bins":50,  "Latex":"Track $\\chi^2_{r\\phi}$ / dof",   "SubFolder":"Trk/", "Loc":"Chi2rphi", "PUsplit":True,  "PDGIDSplit":False,  "match":"trk_matchtp_chi2rphi"},
             "chi2rz_dof" :   {"dataframe":track_data_frame, "xrange":(0,40),         "density":True, "log":True,  "lloc":"upper right",  "bins":50,  "Latex":"Track $\\chi^2_{rz}$ / dof",      "SubFolder":"Trk/", "Loc":"Chi2rz",   "PUsplit":True,  "PDGIDSplit":False,  "match":"trk_matchtp_chi2rz", },
             "trk_bendchi2" : {"dataframe":track_data_frame, "xrange":(0,50 ),        "density":True, "log":True,  "lloc":"upper right",  "bins":50,  "Latex":"Track $\\chi^2_{bend}$",         "SubFolder":"Trk/", "Loc":"Chi2bend", "PUsplit":True,  "PDGIDSplit":False,  "match":"trk_matchtp_bendchi2"},
             "trk_nstub"    : {"dataframe":track_data_frame, "xrange":(3,8),          "density":True, "log":False, "lloc":"upper right",  "bins":0,   "Latex":"Track # Stubs",                  "SubFolder":"Trk/", "Loc":"Nstub",    "PUsplit":True,  "PDGIDSplit":False },
             "nlay_miss"    : {"dataframe":track_data_frame, "xrange":(0,5),          "density":True, "log":False, "lloc":"upper right",  "bins":0,   "Latex":"Track # Missing Internal Stubs", "SubFolder":"Trk/", "Loc":"NLayMiss", "PUsplit":True,  "PDGIDSplit":False },
             "trk_MVA1"     : {"dataframe":track_data_frame, "xrange":(0,1),          "density":True, "log":True,  "lloc":"upper center",  "bins":50,  "Latex":"Track BDT Score",                "SubFolder":"Trk/", "Loc":"MVA",      "PUsplit":True,  "PDGIDSplit":False,  "match":"trk_matchtp_MVA1"},
             "trk_TanL"     : {"dataframe":track_data_frame, "xrange":(-5.5,5.5),     "density":True, "log":True,  "lloc":"lower center",  "bins":50,  "Latex":"Track Tan($\\lambda$)",           "SubFolder":"Trk/", "Loc":"TanL",     "PUsplit":True,  "PDGIDSplit":False},
             "trk_nPSstub_hitpattern"      : {"dataframe":track_data_frame, "xrange":(0,7),          "density":True, "log":False, "lloc":"upper right",  "bins":0,   "Latex":"Track # PS Stubs",                  "SubFolder":"Trk/", "Loc":"NPSstub",    "PUsplit":True,  "PDGIDSplit":False },
             "trk_n2Sstub_hitpattern"      : {"dataframe":track_data_frame, "xrange":(0,7),          "density":True, "log":False, "lloc":"upper right",  "bins":0,   "Latex":"Track # 2S Stubs",                  "SubFolder":"Trk/", "Loc":"N2Sstub",    "PUsplit":True,  "PDGIDSplit":False },
             "trk_nLostPSstub_hitpattern"  : {"dataframe":track_data_frame, "xrange":(0,7),          "density":True, "log":False, "lloc":"upper right",  "bins":0,   "Latex":"Track # Missing PS Stubs",          "SubFolder":"Trk/", "Loc":"NLayPSMiss", "PUsplit":True,  "PDGIDSplit":False },
             "trk_nLost2Sstub_hitpattern"  : {"dataframe":track_data_frame, "xrange":(0,7),          "density":True, "log":False, "lloc":"upper right",  "bins":0,   "Latex":"Track # Missing 2S Stubs",          "SubFolder":"Trk/", "Loc":"NLay2SMiss", "PUsplit":True,  "PDGIDSplit":False },
             "trk_nLoststub_V1_hitpattern" : {"dataframe":track_data_frame, "xrange":(0,7),          "density":True, "log":False, "lloc":"upper right",  "bins":0,   "Latex":"Track # Missing Internal V1 Stubs", "SubFolder":"Trk/", "Loc":"NLayMissV1", "PUsplit":True,  "PDGIDSplit":False },
             "trk_nLoststub_V2_hitpattern" : {"dataframe":track_data_frame, "xrange":(0,7),          "density":True, "log":False, "lloc":"upper right",  "bins":0,   "Latex":"Track # Missing Internal V2 Stubs", "SubFolder":"Trk/", "Loc":"NLayMissV2", "PUsplit":True,  "PDGIDSplit":False },
             "tp_eta" :       {"dataframe":tp_data_frame,    "xrange":(-2.7,2.7),     "density":True, "log":False, "lloc":"upper right",  "bins":50,  "Latex":"TP $\\eta$",                      "SubFolder":"TP/",  "Loc":"Eta",      "PUsplit":False, "PDGIDSplit":False},
             "tp_pt" :        {"dataframe":tp_data_frame,    "xrange":(1,10),         "density":True, "log":True,  "lloc":"upper right",  "bins":50,  "Latex":"TP $p_T$",                       "SubFolder":"TP/",  "Loc":"PtLow",    "PUsplit":False, "PDGIDSplit":False},
             "tp_z0" :        {"dataframe":tp_data_frame,    "xrange":(-20,20),       "density":True, "log":False, "lloc":"upper right",  "bins":50,  "Latex":"TP $z_0$",                       "SubFolder":"TP/",  "Loc":"Z0",       "PUsplit":False, "PDGIDSplit":False}             
             }

fake_tracks = track_data_frame["trk_fake"] == 0
real_tracks = track_data_frame["trk_fake"] != 0
PV_tracks = track_data_frame["trk_fake"] == 1
PU_tracks = track_data_frame["trk_fake"] == 2

PV_TP = tp_data_frame["tp_eventid"] == 0
PU_TP = tp_data_frame["tp_eventid"] != 0

E_tracks = (abs(track_data_frame["trk_matchtp_pdgid"]) == 11)
M_tracks = (abs(track_data_frame["trk_matchtp_pdgid"]) == 13)
H_tracks = ((abs(track_data_frame["trk_matchtp_pdgid"]) != 11) & (abs(track_data_frame["trk_matchtp_pdgid"]) != 13))

for key, value in plot_dict.items():
    plt.close()
    plt.clf()
    #print(value["dataframe"].head())
    figure=plot_variable(value["dataframe"][key],value["Latex"],"",
                         xrange=value["xrange"],density=value["density"],log=value["log"],
                          bins = value["bins"])
    plt.savefig("%s%s%sDistribution.png" % (outputFolder, value["SubFolder"], value["Loc"]))

    if value["PUsplit"]:
        splitfolder = "PUSplit/"
        plt.close()
        plt.clf()
        figure = plot_split_variable(value["dataframe"][key],value["Latex"],
                                    [real_tracks,fake_tracks],["Real Tracks","Fake Tracks"],"",
                                    xrange=value["xrange"],density=value["density"],log=value["log"],
                                    lloc = value["lloc"], bins = value["bins"])
        plt.savefig("%s%s%s%sSplitDistribution.png" % (outputFolder,value["SubFolder"],splitfolder, value["Loc"]))


        plt.close()
        plt.clf()
        figure = plot_split_variable(value["dataframe"][key],value["Latex"],
                                    [PV_tracks,fake_tracks,PU_tracks],
                                    ["PV Tracks","Fake Tracks", "PU Tracks"],"",
                                    xrange=value["xrange"],density=value["density"],log=value["log"],
                                    lloc = value["lloc"], bins = value["bins"])
        plt.savefig("%s%s%s%sPUSplitDistribution.png" % (outputFolder, value["SubFolder"],splitfolder, value["Loc"]))

    if value["PDGIDSplit"]:
        splitfolder = "PDGIDSplit/"
        plt.close()
        plt.clf()
        figure = plot_multiple_variable([value["dataframe"][value["match"]][E_tracks],
                                      value["dataframe"][value["match"]][M_tracks],
                                      value["dataframe"][value["match"]][H_tracks],
                                      value["dataframe"][key][fake_tracks]],value["Latex"],
                                    ["e Tracks","$\\mu$ Tracks", "h Tracks","Fake Tracks"],"",
                                    xrange=value["xrange"],density=value["density"],log=value["log"],
                                    lloc = value["lloc"], bins = value["bins"])
        plt.savefig("%s%s%s%sPDGIDSplitDistribution.png" % (outputFolder, value["SubFolder"],splitfolder,value["Loc"]))

##### 2D Plots ######

plt.close()
plt.clf()
figure = plot_2d(track_data_frame["trk_z0"][real_tracks],track_data_frame["trk_eta"][real_tracks],(-15,15),(-2.4,2.4),"Track $z_0$ [cm]","Track $\\eta$","Distribution for Real Tracks")
plt.savefig("%s/twoD/z0etareals_2D.png" % outputFolder)

plt.close()
plt.clf()
figure = plot_2d(track_data_frame["trk_z0"][fake_tracks],track_data_frame["trk_eta"][fake_tracks],(-15,15),(-2.4,2.4),"Track $z_0$ [cm]","Track $\\eta$","Distribution for Fake Tracks")
plt.savefig("%s/twoD/z0etafakes_2D.png" % outputFolder)

plt.close()
plt.clf()
figure = plot_2d(track_data_frame["trk_pt"][real_tracks],track_data_frame["trk_eta"][real_tracks],(2,128),(-2.4,2.4),"Track $p_T$ [GeV]","Track $\\eta$","Distribution for Real Tracks")
plt.savefig("%s/twoD/ptetareals_2D.png" % outputFolder)

plt.close()
plt.clf()
figure = plot_2d(track_data_frame["trk_pt"][fake_tracks],track_data_frame["trk_eta"][fake_tracks],(2,128),(-2.4,2.4),"Track $p_T$ [GeV]","Track $\\eta$","Distribution for Fake Tracks")
plt.savefig("%s/twoD/ptetafakes_2D.png" % outputFolder)

plt.close()
plt.clf()
figure = plot_2d(track_data_frame["trk_eta"][real_tracks],track_data_frame["trk_MVA1"][real_tracks],(-2.4,2.4),(0,1),"Track $\\eta$","Track BDT Score","Distribution for Real Tracks")
plt.savefig("%s/twoD/mvaetareals_2D.png" % outputFolder)

plt.close()
plt.clf()
figure = plot_2d(track_data_frame["trk_eta"][fake_tracks],track_data_frame["trk_MVA1"][fake_tracks],(-2.4,2.4),(0,1),"Track $\\eta$","Track BDT Score","Distribution for Fake Tracks")
plt.savefig("%s/twoD/mvaetafakes_2D.png" % outputFolder)

