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


"python track_efficiency.py NoDegredation_Zp_TrackNtuple.root Test_Zp True True 1"
"python track_efficiency.py RootFilename.root name_of_dataset save_arrays run_BDT number_of_batches "
rootdir = "/home/cb719/Documents/Datasets/TrackDatasets/NoDegredation/"
#rootdir = "/home/cb719/Documents/Datasets/TrackDatasets/"
f = sys.argv[1]
name = sys.argv[2]
save = sys.argv[3]
runBDT = sys.argv[4]

outputFolder = name + "/Plots/"
os.system("mkdir "+name)
os.system("mkdir "+name+ "/Plots")
os.system("mkdir "+name+ "/Efficiencies")

track_data_frame = pd.DataFrame()
tp_data_frame = pd.DataFrame()
match_data_frame = pd.DataFrame()

max_batch = int(sys.argv[5])

from Models.GBDTTrackQualityModel import XGBoostClassifierModel
NDmodel_name = "Degredation9"
NDmodel = XGBoostClassifierModel(NDmodel_name)
NDmodel.load_model("../../Projects/"+NDmodel_name+"/Models/",NDmodel_name)

track_feature_list = ['trk_pt', 'trk_eta', 'trk_phi', 
                      'trk_z0', 'trk_chi2rphi', 
                      'trk_chi2rz', 'trk_bendchi2', 'trk_MVA1',
                      'trk_nstub', 'trk_fake','trk_hitpattern','trk_genuine']

match_feature_list = ['matchtrk_pt', 'matchtrk_eta', 'matchtrk_phi', 
                      'matchtrk_z0', 'matchtrk_chi2rphi', 
                      'matchtrk_chi2rz', 'matchtrk_bendchi2', 'matchtrk_MVA1',
                      'matchtrk_nstub','matchtrk_hitpattern']

tp_feature_list = ['tp_pt','tp_eta', 'tp_phi', 'tp_z0','tp_pdgid', 'tp_nstub', 'tp_eventid', 'tp_charge', 'tp_nmatch',
                   'tp_dxy','tp_d0']

if save == "True":
    os.system("mkdir "+name+ "/Arrays")


    tracks = uproot.open(rootdir+f+":L1TrackNtuple/eventTree", num_workers=8)

    track_data_frame = save_branches(tracks,track_data_frame,track_feature_list,max_batch=max_batch,name="tracks")

    track_data_frame["nlay_miss"] = track_data_frame["trk_hitpattern"].apply(set_nlaymissinterior)

    track_data_frame["chi2rphi_dof"] = track_data_frame["trk_chi2rphi"]/(track_data_frame["trk_nstub"] - 2)
    track_data_frame["chi2rz_dof"] = track_data_frame["trk_chi2rz"]/(track_data_frame["trk_nstub"] - 2)

    track_data_frame["TanL"] = track_data_frame["trk_eta"].apply(tanl)

    track_data_frame["AbsZ"] = track_data_frame["trk_z0"].apply(abs)
    track_data_frame["RescaledAbsZ"] = track_data_frame["AbsZ"]/(2**(trackword_config['Z0']['nbits'] - 1)*trackword_config['Z0']['granularity'])

    track_data_frame.loc[:,"bit_bendchi2"] = track_data_frame["trk_bendchi2"].apply(np.digitize,bins=trackword_config["Bendchi2"]["bins"]) - 1
    track_data_frame.loc[:,"bit_chi2rphi"] = track_data_frame["chi2rphi_dof"].apply(np.digitize,bins=trackword_config["Chi2rphi"]["bins"]) - 1
    track_data_frame.loc[:,"bit_chi2rz"]   = track_data_frame["chi2rz_dof"].apply(np.digitize,bins=trackword_config["Chi2rz"]["bins"]) - 1

    match_data_frame = save_branches(tracks,match_data_frame,match_feature_list,max_batch=max_batch,name="match tracks")

    match_data_frame["nlay_miss"] = match_data_frame["matchtrk_hitpattern"].apply(set_nlaymissinterior)

    match_data_frame["chi2rphi_dof"] = match_data_frame["matchtrk_chi2rphi"]/(match_data_frame["matchtrk_nstub"] - 2)
    match_data_frame["chi2rz_dof"] = match_data_frame["matchtrk_chi2rz"]/(match_data_frame["matchtrk_nstub"] - 2)
    match_data_frame["trk_nstub"] = match_data_frame["matchtrk_nstub"]

    match_data_frame["TanL"] = match_data_frame["matchtrk_eta"].apply(tanl)

    match_data_frame["AbsZ"] = match_data_frame["matchtrk_z0"].apply(abs)
    match_data_frame["RescaledAbsZ"] = match_data_frame["AbsZ"]/(2**(trackword_config['Z0']['nbits'] - 1)*trackword_config['Z0']['granularity'])

    match_data_frame.loc[:,"bit_bendchi2"] = match_data_frame["matchtrk_bendchi2"].apply(np.digitize,bins=trackword_config["Bendchi2"]["bins"]) - 1
    match_data_frame.loc[:,"bit_chi2rphi"] = match_data_frame["chi2rphi_dof"].apply(np.digitize,bins=trackword_config["Chi2rphi"]["bins"]) - 1
    match_data_frame.loc[:,"bit_chi2rz"]   = match_data_frame["chi2rz_dof"].apply(np.digitize,bins=trackword_config["Chi2rz"]["bins"]) - 1


    if runBDT:
        
        y_predict_proba = model.model.predict_proba(track_data_frame[model.training_features].to_numpy())[:,1]
        y_predict_binned = np.digitize(y_predict_proba,trackword_config["MVA1"]["bins"])
        y_predict_binned = (y_predict_binned - 1) / 8

        track_data_frame["BDT_output"] = y_predict_proba
        track_data_frame["BDT_binned_output"] = y_predict_binned

        y_predict_proba = model.model.predict_proba(match_data_frame[model.training_features].to_numpy())[:,1]
        y_predict_binned = np.digitize(y_predict_proba,trackword_config["MVA1"]["bins"])
        y_predict_binned = (y_predict_binned - 1) / 8

        match_data_frame["BDT_output"] = y_predict_proba
        match_data_frame["BDT_binned_output"] = y_predict_binned

    track_data_frame.to_pickle(name+"/Arrays/track_data_frame.pkl") 

    tp_data_frame = save_branches(tracks,tp_data_frame,tp_feature_list,max_batch=max_batch,name="tps")

    # Save, can modify these locations as needed

    tp_data_frame.to_pickle(name+"/Arrays/tp_data_frame.pkl") 
    match_data_frame.to_pickle(name+"/Arrays/match_data_frame.pkl") 

if save == "False":

    track_data_frame = pd.read_pickle(name+"/Arrays/track_data_frame.pkl") 
    tp_data_frame = pd.read_pickle(name+"/Arrays/tp_data_frame.pkl") 
    match_data_frame = pd.read_pickle(name+"/Arrays/match_data_frame.pkl") 

    if runBDT:
        y_predict_proba = NDmodel.model.predict_proba(track_data_frame[NDmodel.training_features].to_numpy())[:,1]
        y_predict_binned = np.digitize(y_predict_proba,trackword_config["MVA1"]["bins"])
        y_predict_binned = (y_predict_binned - 1) / 8

        track_data_frame["NDBDT_output"] = y_predict_proba
        track_data_frame["NDBDT_binned_output"] = y_predict_binned

        y_predict_proba = NDmodel.model.predict_proba(match_data_frame[NDmodel.training_features].to_numpy())[:,1]
        y_predict_binned = np.digitize(y_predict_proba,trackword_config["MVA1"]["bins"])
        y_predict_binned = (y_predict_binned - 1) / 8

        match_data_frame["NDBDT_output"] = y_predict_proba
        match_data_frame["NDBDT_binned_output"] = y_predict_binned

        ############################################


NDBDT_threshold = 0.65

chi2_cut_tracks = track_data_frame.query('chi2rphi_dof < 5 & chi2rz_dof < 20 & trk_bendchi2 < 2.25')
NDMVA_cut_tracks = track_data_frame.query('NDBDT_output > ' + str(NDBDT_threshold))


plot_dict = {"pt" :       {"xrange":[0,100],        "yrange":[0.0,1,0,0.2],  "bins":50,  "Latex":"$p_T$ [GeV]", "Loc":"Pt",   },
             "eta" :      {"xrange":[-2.4,2.4],     "yrange":[0.0,1,0,0.06], "bins":50,  "Latex":"$\\eta$",     "Loc":"Eta",  },
             #"phi" :      {"xrange":[-np.pi,np.pi], "yrange":[0,1,0,0.06], "bins":50, "Latex":"$\\phi_0$",   "Loc":"Phi0", },
             #"z0" :       {"xrange":[-20,20],       "yrange":[0,1,0,0.2],  "bins":50,  "Latex":"$z_0$ [cm]",  "Loc":"Z0",   }}
            }
for key, value in plot_dict.items():
      print("Calculating " + key + " efficiencies") 
      baseline_numbers = calculate_var_bins(track_data_frame,tp_data_frame,match_data_frame,second_query_func = ('matchtrk_pt > 0'),
                                              variable=key,var_range=value['xrange'],n_bins=value['bins'])
      chi2_numbers = calculate_var_bins(chi2_cut_tracks,tp_data_frame,match_data_frame,second_query_func = ('chi2rphi_dof < 5 & chi2rz_dof < 20 & matchtrk_bendchi2 < 2.25'),
                                          variable=key,var_range=value['xrange'],n_bins=value['bins'])
      NDMVA_numbers = calculate_var_bins(NDMVA_cut_tracks,tp_data_frame,match_data_frame,second_query_func = ('NDBDT_output > '+ str(NDBDT_threshold)),
                                          variable=key,var_range=value['xrange'],n_bins=value['bins'])

      plot_eff_fake_bins([baseline_numbers,chi2_numbers,NDMVA_numbers],["Baseline","$\\chi^2$ Cuts","BDT > "+str(NDBDT_threshold)],typeset_var=value['Latex'],filename=value['Loc'],ylims=value['yrange'],outputFolder=outputFolder)

baseline_numbers = calculate_track_numbers(track_data_frame,tp_data_frame,match_data_frame,second_query_func = ('matchtrk_pt > 0'))
chi2_numbers = calculate_track_numbers(chi2_cut_tracks,tp_data_frame,match_data_frame,second_query_func = ('chi2rphi_dof < 5 & chi2rz_dof < 20 & matchtrk_bendchi2 < 2.25'))
MVA_numbers = calculate_track_numbers(NDMVA_cut_tracks,tp_data_frame,match_data_frame,second_query_func = ('NDBDT_output > '+ str(NDBDT_threshold)))

print_efficiencies("Baseline", baseline_numbers)
print_efficiencies("Chi2", chi2_numbers)
print_efficiencies("BDT > "+str(NDBDT_threshold),MVA_numbers)

# BDT_thresholds = np.linspace(0,1,20)
# NDBDT_effs = []
# baseline_effs = convert_to_efficiencies(calculate_track_numbers(track_data_frame,tp_data_frame,match_data_frame,second_query_func = ('matchtrk_pt > 0')))
# chi2_effs = convert_to_efficiencies(calculate_track_numbers(chi2_cut_tracks,tp_data_frame,match_data_frame,second_query_func = ('chi2rphi_dof < 5 & chi2rz_dof < 20 & matchtrk_bendchi2 < 2.25')))

# for threshold in BDT_thresholds:
#     NDMVA_cut_tracks = track_data_frame.query('NDBDT_output > ' + str(threshold))
#     MVA3_cut_tracks = track_data_frame.query('BDT3_output > ' + str(threshold))
#     MVA9_cut_tracks = track_data_frame.query('BDT9_output > ' + str(threshold))
#     NDBDT_effs.append(convert_to_efficiencies(calculate_track_numbers(NDMVA_cut_tracks,tp_data_frame,match_data_frame,second_query_func = ('NDBDT_output > '+ str(threshold)))))
#     BDT3_effs.append(convert_to_efficiencies(calculate_track_numbers(MVA3_cut_tracks,tp_data_frame,match_data_frame,second_query_func = ('BDT3_output > '+ str(threshold)))))
#     BDT9_effs.append(convert_to_efficiencies(calculate_track_numbers(MVA9_cut_tracks,tp_data_frame,match_data_frame,second_query_func = ('BDT9_output > '+ str(threshold)))))



# plot_eff_fake_curve([BDT3_effs,NDBDT_effs],["Current CMSSW BDT","New Training BDT"],outputFolder=outputFolder)

# with open(name+ "/Efficiencies/"+NDmodel_name+"BDT_effs.pkl", 'wb') as f:
#     pickle.dump(NDBDT_effs, f)
# with open(name+ "/Efficiencies/"+model3_name+"BDT_effs.pkl", 'wb') as f:
#     pickle.dump(BDT3_effs, f)
# with open(name+ "/Efficiencies/"+model9_name+"BDT_effs.pkl", 'wb') as f:
#     pickle.dump(BDT9_effs, f)
# with open(name+ "/Efficiencies/NewKF_baseline_effs.pkl", 'wb') as f:
#     pickle.dump(baseline_effs, f)
# with open(name+ "/Efficiencies/NewKF_chi2_effs.pkl", 'wb') as f:
#    pickle.dump(chi2_effs, f)