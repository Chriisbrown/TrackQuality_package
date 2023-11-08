import uproot
import os
import sys
from funcs import *
from plotting import *
import matplotlib.pyplot as plt
import pickle
import helixplotter
import constants

"python event_runner.py RootFilename.root name_of_dataset save_arrays"


rootdir = "/home/cb719/Documents/Datasets/TrackDatasets/"
f = sys.argv[1]
name = sys.argv[2]
save = True

useTPMET = True

outputFolder = name + "/Plots/Vertex"
METoutputFolder = name + "/Plots/MET"

os.system("mkdir "+name)
os.system("mkdir "+name+ "/Plots")
os.system("mkdir "+name+ "/Arrays")
os.system("mkdir "+name+ "/Plots/Vertex")
os.system("mkdir "+name+ "/Plots/MET")
os.system("mkdir "+name+ "/Arrays/Vertex")
os.system("mkdir "+name+ "/Arrays/MET")


chunkread = 5000
batch_num = 0
event_num = 0

events = uproot.open(rootdir+f+':L1TrackNtuple/eventTree',num_workers=8)
num_events = events.num_entries

FHs = {"Baseline" : {"array" : np.ndarray([num_events]), "Save" : "Baseline_vtx.npy", "Latex": "Reco $Z_{0,Tracks}^{PV}$",                    "Colours": "red"},
       "Baseline3GeV" : {"array" : np.ndarray([num_events]), "Save" : "Baseline3Gev_vtx.npy", "Latex": "Reco $Z_{0,Tracks}^{PV}$ $p_T > 3$ GeV",                    "Colours": "blue"},
       "Baseline128GeV" : {"array" : np.ndarray([num_events]), "Save" : "Baseline128Gev_vtx.npy", "Latex": "Reco $Z_{0,Tracks}^{PV}$ $2 < p_T < 128$ GeV",                    "Colours": "black"},
       "OnlyPVs"  : {"array" : np.ndarray([num_events]), "Save" : "True_vtx.npy",     "Latex": "True $Z_{0,Tracks}^{PV}$",                    "Colours": "black"},
       "Chi2" :     {"array" : np.ndarray([num_events]), "Save" : "Chi2_vtx.npy",     "Latex": "Reco $Z_{0,Tracks}^{PV}$ $\\chi^2$ Selection","Colours": "blue"},
       "TQ" :       {"array" : np.ndarray([num_events]), "Save" : "TQ_vtx.npy",       "Latex": "Reco $Z_{0,Tracks}^{PV}$ BDT Selection",      "Colours": "green"},
       "Eta" :      {"array" : np.ndarray([num_events]), "Save" : "Eta_vtx.npy",      "Latex": "Reco $Z_{0,Tracks}^{PV}$ $\\eta$ Weighting" , "Colours": "orange"},
       "Combined" : {"array" : np.ndarray([num_events]), "Save" : "Combined_vtx.npy", "Latex": "Reco $Z_{0,Tracks}^{PV}$ Combined Weighting", "Colours": "purple"}}

MET = {"Baseline" : {"array" : np.ndarray([num_events]), "Save" : "Baseline_met.npy", "Latex": "Reco $E_{T,Tracks}^{miss}$ No Selection",               "Colours": "red"},
       "OnlyZ0" :   {"array" : np.ndarray([num_events]), "Save" : "OnlyZ0_met.npy",   "Latex": "Reco $E_{T,Tracks}^{miss}$             ",  "Colours": "pink"},
       "OnlyPVs"  : {"array" : np.ndarray([num_events]), "Save" : "True_met.npy",     "Latex": "True $E_{T,Tracks}^{miss}$ ",              "Colours": "black"},
       "Chi2" :     {"array" : np.ndarray([num_events]), "Save" : "Chi2_met.npy",     "Latex": "Reco $E_{T,Tracks}^{miss}$ $\\chi^2$ Cuts","Colours": "blue"},
       "TQ" :       {"array" : np.ndarray([num_events]), "Save" : "TQ_met.npy",       "Latex": "Reco $E_{T,Tracks}^{miss}$ BDT Cuts",      "Colours": "green"}}

num_threshold = 10
thresholds = [str(i/num_threshold) for i in range(0,num_threshold)]
predictedZ0_MVA = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
predictedMET_MVA = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}

MCs = {"Vertex" :   {"array" : np.ndarray([num_events]), "Save" : "Actual_Vtx.npy",   "Latex": "Actual Vertex"},
       "MET" :   {"array" : np.ndarray([num_events]), "Save" : "Actual_MET.npy",   "Latex": "Actual $E_T^{miss}$"}}


if save:
    for batch in events.iterate(step_size=chunkread, library='numpy'):
        #if batch_num > 9:
        #        break
        for ievt in range(len(batch["trk_pt"])):
            event = {i: batch[i][ievt] for i in batch.keys()}
            ########### FASTHISTOs  ###########
            baselineFH_vtx = predictFastHisto(batch['trk_z0'][ievt],
                                            batch['trk_pt'][ievt],
                                            num_vertices=1,
                                            weighting=linear_res_function(batch['trk_z0'][ievt]))

            baselineFH3GeV_vtx = predictFastHisto(batch['trk_z0'][ievt],
                                            batch['trk_pt'][ievt],
                                            num_vertices=1,
                                            weighting=MinPt(batch['trk_pt'][ievt],3))
            
            baselineFH128GeV_vtx = predictFastHisto(batch['trk_z0'][ievt],
                                            batch['trk_pt'][ievt],
                                            num_vertices=1,
                                            weighting=MaxPt(batch['trk_pt'][ievt],128))

            Chi2FH_vtx = predictFastHisto(batch['trk_z0'][ievt],
                                        batch['trk_pt'][ievt],
                                        num_vertices=1,
                                        weighting=chi_res_function(batch['trk_chi2rphi'][ievt],
                                                                    batch['trk_chi2rz'][ievt],
                                                                    batch['trk_bendchi2'][ievt],
                                                                    batch['trk_nstub'][ievt])
                                        )

            TQFH_vtx = predictFastHisto(batch['trk_z0'][ievt],
                                        batch['trk_pt'][ievt],
                                        num_vertices=1,
                                        weighting=mva_res_function(batch['trk_MVA1'][ievt]))

            EtaFH_vtx = predictFastHisto(batch['trk_z0'][ievt],
                                        batch['trk_pt'][ievt],
                                        num_vertices=1,
                                        weighting=eta_res_function(batch['trk_eta'][ievt]))

            CombinedFH_vtx = predictFastHisto(batch['trk_z0'][ievt],
                                            batch['trk_pt'][ievt],
                                            num_vertices=1,
                                            weighting=comb_res_function(batch['trk_MVA1'][ievt],
                                                                        batch['trk_eta'][ievt]))

            OnlyPVFH_vtx = predictFastHisto(batch['trk_z0'][ievt],
                                            batch['trk_pt'][ievt],
                                            num_vertices=1,
                                            weighting=OnlyPV_function(batch['trk_fake'][ievt]))

            for i in range(0,num_threshold):
                FH_MVA = predictFastHisto(batch['trk_z0'][ievt],
                                          batch['trk_pt'][ievt],
                                          weighting=mva_res_function(batch['trk_MVA1'][ievt],
                                          threshold=i/num_threshold))
                predictedZ0_MVA[str(i/num_threshold)].append(FH_MVA)

            
            baseline_assoc = FastHistoAssoc(baselineFH_vtx,
                                            batch['trk_z0'][ievt],
                                            batch['trk_eta'][ievt])

            no_assoc = np.ones_like(baseline_assoc,dtype=bool)

            chi2_selection =  ((batch['trk_chi2rphi'][ievt] < 12 )
                               & (batch['trk_chi2rz'][ievt] < 10)
                               & (batch['trk_bendchi2'][ievt] < 4)
                               & (batch['trk_nstub'][ievt] > 3)
                               & baseline_assoc)

            mva_selection = ((batch['trk_MVA1'][ievt] > 0.7 ) & baseline_assoc)

            

            PV_selection = (batch['trk_fake'][ievt] == 1 )

            MET["Baseline"]["array"][event_num] = predictMET(batch['trk_pt'][ievt],
                                                             batch['trk_phi'][ievt],
                                                             no_assoc)

            MET["OnlyZ0"]["array"][event_num] =   predictMET(batch['trk_pt'][ievt],
                                                             batch['trk_phi'][ievt],
                                                             baseline_assoc)

            MET["OnlyPVs"]["array"][event_num] =   predictMET(batch['trk_pt'][ievt],
                                                             batch['trk_phi'][ievt],
                                                             PV_selection)

            MET["Chi2"]["array"][event_num] =   predictMET(batch['trk_pt'][ievt],
                                                             batch['trk_phi'][ievt],
                                                             chi2_selection)

            MET["TQ"]["array"][event_num] =   predictMET(batch['trk_pt'][ievt],
                                                             batch['trk_phi'][ievt],
                                                             mva_selection)
            
            for i in range(0,num_threshold):
                mva_selection = ((batch['trk_MVA1'][ievt] > i/num_threshold ) & baseline_assoc)
                MET_MVA =   predictMET(batch['trk_pt'][ievt],
                                       batch['trk_phi'][ievt],
                                       mva_selection)
                predictedMET_MVA[str(i/num_threshold)].append(MET_MVA)

            if useTPMET:
                TP_selection = ((batch['tp_pt'][ievt] >= 2 )
                            & (abs(batch['tp_eta'][ievt]) <= 2.4 )
                            & (batch['tp_nstub'][ievt] >= 4 )
                            & (abs(batch['tp_z0'][ievt]) <= 30 ) 
                            & (batch['tp_eventid'][ievt] == 0 )
                            & (batch['tp_charge'][ievt]  != 0))


                MCs["MET"]["array"][event_num] = predictMET(batch['tp_pt'][ievt],
                                                            batch['tp_phi'][ievt],
                                                            TP_selection)



            #print(chi2_selection)
            #print(batch[0]['trk_pt'][ievt].to_numpy()[chi2_selection])
            #print("=================================================")
            #print(gen_selection)
            #print(batch[4]['gen_pt'][ievt])
            #print(MCs["MET"]["array"][event_num])
            #print(MET["TQ"]["array"][event_num],MET["OnlyPVs"]["array"][event_num],MET["OnlyZ0"]["array"][event_num],MET["Chi2"]["array"][event_num])


            MCs['Vertex']["array"][event_num] = (batch['pv_MC'][ievt][0])
            FHs['Baseline']["array"][event_num] = baselineFH_vtx[0]
            FHs['Baseline3GeV']["array"][event_num] = baselineFH3GeV_vtx[0]
            FHs['Baseline128GeV']["array"][event_num] = baselineFH128GeV_vtx[0]
            FHs['Chi2']["array"][event_num]     = Chi2FH_vtx[0]
            FHs['TQ']["array"][event_num]       = TQFH_vtx[0]
            FHs['Eta']["array"][event_num]      = EtaFH_vtx[0]
            FHs['Combined']["array"][event_num] = CombinedFH_vtx[0]
            FHs['OnlyPVs']["array"][event_num]  = OnlyPVFH_vtx[0]

            ######## MET ############

            if (ievt % 1000 == 0):
                print("Event: ", ievt, " out of ", len(batch["trk_pt"]))

            event_num += 1
            ##############################################################

        batch_num += 1

        #Add batch dataframe to larger dataframe, only defined branches are added to save memory
        print(batch_num," out of: ", len(events))

    # Save, can modify these locations as needed

    filehandler = open(name+"/Arrays/Vertex/MVA_dict", 'wb')
    pickle.dump(predictedZ0_MVA, filehandler)

    for FH in FHs.keys():
        np.save(name+"/Arrays/Vertex/"+FHs[FH]["Save"],FHs[FH]["array"])

    for MC in MCs:
        np.save(name+"/Arrays/"+MCs[MC]["Save"],MCs[MC]["array"])

    filehandler = open(name+"/Arrays/MET/MVAMET_dict", 'wb')
    pickle.dump(predictedMET_MVA, filehandler)

    for iMET in MET.keys():
        np.save(name+"/Arrays/MET/"+MET[iMET]["Save"],MET[iMET]["array"])

if not save:

    for FH in FHs.keys():
        FHs[FH]["array"] = np.load(name+"/Arrays/Vertex/"+FHs[FH]["Save"])

    for iMET in MET.keys():
        MET[iMET]["array"] = np.load(name+"/Arrays/MET/"+MET[iMET]["Save"])

    for MC in MCs:
        MCs[MC]["array"] = np.load(name+"/Arrays/"+MCs[MC]["Save"])

    with open(name+"/Arrays/Vertex/MVA_dict", 'rb') as f:
        predictedZ0_MVA = pickle.load(f)

    with open(name+"/Arrays/MET/MVAMET_dict", 'rb') as f:
        predictedMET_MVA = pickle.load(f)



predictedZ0_MVA_array = {key: value for key, value in zip(thresholds, [np.zeros(1) for i in range(0,num_threshold)])}

predictedZ0_MVA_RMS_array = np.zeros([num_threshold])
predictedZ0_MVA_Quartile_array = np.zeros([num_threshold])
predictedZ0_MVA_Centre_array = np.zeros([num_threshold])

MVA_histos = []
MVA_log_histos = []

for i in range(0,num_threshold):
    z0_MVA_array  = np.concatenate(predictedZ0_MVA[str(i/num_threshold)]).ravel()
    predictedZ0_MVA_array[str(i/num_threshold)] = z0_MVA_array
    Diff = MCs['Vertex']["array"] - z0_MVA_array

    predictedZ0_MVA_RMS_array[i] = np.sqrt(np.mean(Diff**2))
    qMVA = np.percentile(Diff,[32,50,68])

    predictedZ0_MVA_Quartile_array[i] = qMVA[2] - qMVA[0]
    predictedZ0_MVA_Centre_array[i] = qMVA[1]

    hist,bin_edges = np.histogram((MCs['Vertex']["array"] - z0_MVA_array),bins=50,range=(-1,1))
    hist_log,bin_edges = np.histogram((MCs['Vertex']["array"] - z0_MVA_array),bins=50,range=(-1*constants.max_z0,constants.max_z0))
    MVA_histos.append(hist)
    MVA_log_histos.append(hist_log)

Quartilethreshold_choice = str(np.argmin(predictedZ0_MVA_Quartile_array)/num_threshold)
RMSthreshold_choice= str(np.argmin(predictedZ0_MVA_RMS_array)/num_threshold)

MVA_bestQ_array = predictedZ0_MVA_array[Quartilethreshold_choice]
MVA_bestRMS_array = predictedZ0_MVA_array[RMSthreshold_choice]

FHwidths = calc_widths(MCs['Vertex']["array"],(FHs["Baseline"]["array"]))
FH3GeVwidths = calc_widths(MCs['Vertex']["array"],(FHs["Baseline3GeV"]["array"]))
FH128GeVwidths = calc_widths(MCs['Vertex']["array"],(FHs["Baseline128GeV"]["array"]))
FHPVWidths = calc_widths(MCs['Vertex']["array"],(FHs["OnlyPVs"]["array"]))
FHchi2Widths = calc_widths(MCs['Vertex']["array"],(FHs["Chi2"]["array"]))
FHresWidths = calc_widths(MCs['Vertex']["array"],(FHs["Eta"]["array"]))
FHCombWidths = calc_widths(MCs['Vertex']["array"],(FHs["Combined"]["array"]))

plot_FH = True
if plot_FH:
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(24,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    twod_hist = np.stack(MVA_histos, axis=1)
    twod_log_hist = np.stack(MVA_log_histos, axis=1)

    hist2d = ax[0].imshow(twod_hist,cmap=colormap,aspect='auto',extent=[0,1,-1,1])
    hist2d_log = ax[1].imshow(twod_log_hist,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),aspect='auto',cmap=colormap,extent=[0,1,-1*constants.max_z0,constants.max_z0])

    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_xlabel('BDT Threshold',ha="right",x=1)
    ax[0].set_ylabel('Reconstructed $z_{0}^{PV}$ [cm]',ha="right",y=1)
    ax[1].set_xlabel('BDT Threshold',ha="right",x=1)
    ax[1].set_ylabel('Reconstructed $z_{0}^{PV}$ [cm]',ha="right",y=1)

    cbar = plt.colorbar(hist2d , ax=ax[0])
    cbar.set_label('# Events')

    cbar = plt.colorbar(hist2d_log , ax=ax[1])
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/2dhist.pdf" % outputFolder)

    # plt.close()
    # plt.clf()
    # figure=plotz0_residual(MCs['Vertex']["array"],
    #                        [MVA_bestRMS_array,
    #                         FHs["Baseline"]["array"], 
    #                         FHs["Eta"]["array"],
    #                         FHs["OnlyPVs"]["array"],
    #                         FHs["Chi2"]["array"],
    #                         FHs["Combined"]["array"]],
    #                         [FHs["TQ"]["Latex"],
    #                         FHs["Baseline"]["Latex"],
    #                         FHs["Eta"]["Latex"],
    #                         FHs["OnlyPVs"]["Latex"],
    #                         FHs["Chi2"]["Latex"],
    #                         FHs["Combined"]["Latex"]],
    #                         colours=[FHs["TQ"]["Colours"],
    #                                 FHs["Baseline"]["Colours"],
    #                                 FHs["Eta"]["Colours"],
    #                                 FHs["OnlyPVs"]["Colours"],
    #                                 FHs["Chi2"]["Colours"],
    #                                 FHs["Combined"]["Colours"]])
    # plt.savefig("%s/Z0Residual.pdf" % outputFolder)
    # plt.savefig("%s/Z0Residual.png" % outputFolder)

    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    
    ax[0].hist(MCs['Vertex']["array"],bins=50,range=(-1*constants.max_z0,constants.max_z0),histtype="step",
                 linewidth=LINEWIDTH,color = colours[0],
                 label="MC",density=True)
    ax[0].hist(FHs["Baseline"]["array"],bins=50,range=(-1*constants.max_z0,constants.max_z0),histtype="step",
                 linewidth=LINEWIDTH,color = colours[1],
                 label='Baseline FH',density=True)
    
    ax[1].hist(MCs['Vertex']["array"],bins=50,range=(-1*constants.max_z0,constants.max_z0),histtype="step",
                 linewidth=LINEWIDTH,color = colours[0],
                 label="MC",density=True)
    ax[1].hist(FHs["Baseline"]["array"],bins=50,range=(-1*constants.max_z0,constants.max_z0),histtype="step",
                 linewidth=LINEWIDTH,color = colours[1],
                 label='Baseline FH',density=True)
    
    ax[0].grid(True)
    ax[0].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[0].set_ylabel('Events',ha="right",y=1)
    ax[0].set_yscale("log")
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    ax[1].grid(True)
    ax[1].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[1].set_ylabel('Events',ha="right",y=1)
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
    plt.tight_layout()
    plt.savefig("%s/Z0Range.pdf" % outputFolder)
    plt.savefig("%s/Z0Range.png" % outputFolder)

    plt.close()
    plt.clf()
    figure=plotz0_residual(MCs['Vertex']["array"],
                            [FHs["Baseline"]["array"], 
                            FHs["Eta"]["array"]],
                            [FHs["Baseline"]["Latex"],
                            FHs["Eta"]["Latex"]],
                            colours=[FHs["Baseline"]["Colours"],
                                     "black"])
    plt.savefig("%s/Z0EtaResidual.pdf" % outputFolder)
    plt.savefig("%s/Z0EtaResidual.png" % outputFolder)

    plt.close()
    plt.clf()
    figure=plotz0_residual(MCs['Vertex']["array"],
                            [FHs["Baseline"]["array"],
                            MVA_bestRMS_array],
                            [FHs["Baseline"]["Latex"],
                            (FHs["TQ"]["Latex"] + " > " + RMSthreshold_choice)],
                            colours=[FHs["Baseline"]["Colours"],
                                     "black"])
    plt.savefig("%s/Z0BDTResidual.pdf" % outputFolder)
    plt.savefig("%s/Z0BDTResidual.png" % outputFolder)

    plt.close()
    plt.clf()
    figure=plotz0_residual(MCs['Vertex']["array"],
                            [FHs["Baseline"]["array"], 
                            FHs["Chi2"]["array"]],
                            [FHs["Baseline"]["Latex"],
                            FHs["Chi2"]["Latex"]],
                            colours=[FHs["Baseline"]["Colours"],
                                     "black"])
    plt.savefig("%s/Z0Chi2Residual.pdf" % outputFolder)
    plt.savefig("%s/Z0Chi2Residual.png" % outputFolder)

    plt.close()
    plt.clf()
    figure=plotz0_residual(MCs['Vertex']["array"],
                            [FHs["Baseline"]["array"],
                            MVA_bestRMS_array,
                            FHs["Chi2"]["array"]],
                            [FHs["Baseline"]["Latex"],
                            (FHs["TQ"]["Latex"] + " > " + RMSthreshold_choice),
                            FHs["Chi2"]["Latex"]],
                            colours=[FHs["Baseline"]["Colours"],
                                     FHs["TQ"]["Colours"],
                                     FHs["Chi2"]["Colours"]])
    plt.savefig("%s/Z0BDTChi2Residual.pdf" % outputFolder)
    plt.savefig("%s/Z0BDTChi2Residual.png" % outputFolder)

    plt.close()
    plt.clf()
    # figure=plotz0_residual(MCs['Vertex']["array"],
    #                         [FHs["Baseline"]["array"], 
    #                          FHs["Combined"]["array"]],
    #                         [FHs["Baseline"]["Latex"],
    #                          FHs["Combined"]["Latex"]],
    #                         colours=[FHs["Baseline"]["Colours"],
    #                                  "black"])
    # plt.savefig("%s/Z0CombinedResidual.pdf" % outputFolder)
    # plt.savefig("%s/Z0CombinedResidual.png" % outputFolder)


    ######################################################

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual(MCs['Vertex']["array"],
                                        [(MVA_bestRMS_array),
                                        FHs["Baseline"]["array"], 
                                        FHs["Eta"]["array"],
                                        FHs["OnlyPVs"]["array"],
                                        FHs["Chi2"]["array"],
                                        ],#FHs["Combined"]["array"]],
                                       [FHs["TQ"]["Latex"],
                                        FHs["Baseline"]["Latex"],
                                        FHs["Eta"]["Latex"],
                                        FHs["OnlyPVs"]["Latex"],
                                        FHs["Chi2"]["Latex"],
                                        ],#FHs["Combined"]["Latex"]],
                                        colours=[FHs["TQ"]["Colours"],
                                                 FHs["Baseline"]["Colours"],
                                                 FHs["Eta"]["Colours"],
                                                 FHs["OnlyPVs"]["Colours"],
                                                 FHs["Chi2"]["Colours"],
                                                 ])#FHs["Combined"]["Colours"]])
    fig_1.savefig("%s/Z0Residual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0Residual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0Residual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0Residual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual(MCs['Vertex']["array"],
                                        [FHs["Baseline"]["array"], 
                                        FHs["Eta"]["array"]],
                                        [FHs["Baseline"]["Latex"],
                                        FHs["Eta"]["Latex"]],
                                    colours=[FHs["Baseline"]["Colours"],
                                             "black"])
    fig_1.savefig("%s/Z0EtaResidual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0EtaResidual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0EtaResidual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0EtaResidual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual(MCs['Vertex']["array"],
                                        [FHs["Baseline"]["array"]],
                                        [FHs["Baseline"]["Latex"]],
                                    colours=[FHs["Baseline"]["Colours"]])
    fig_1.savefig("%s/Z0BaselineResidual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0BaselineResidual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0BaselineResidual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0BaselineResidual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual(MCs['Vertex']["array"],
                                        [FHs["Baseline"]["array"],
                                         FHs["Baseline3GeV"]["array"]],
                                        [FHs["Baseline"]["Latex"],
                                         FHs["Baseline3GeV"]["Latex"],],
                                    colours=[FHs["Baseline"]["Colours"],
                                             FHs["Baseline3GeV"]["Colours"]])
    fig_1.savefig("%s/Z0Baseline3GevResidual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0Baseline3GeVResidual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0Baseline3GeVResidual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0Baseline3GeVResidual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual(MCs['Vertex']["array"],
                                        [FHs["Baseline"]["array"],
                                         FHs["Baseline128GeV"]["array"]],
                                        [FHs["Baseline"]["Latex"],
                                         FHs["Baseline128GeV"]["Latex"],],
                                    colours=[FHs["Baseline"]["Colours"],
                                             FHs["Baseline128GeV"]["Colours"]])
    fig_1.savefig("%s/Z0Baseline128GevResidual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0Baseline128GeVResidual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0Baseline128GeVResidual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0Baseline128GeVResidual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual(MCs['Vertex']["array"],
                            [FHs["Baseline"]["array"],
                             MVA_bestRMS_array],
                            [FHs["Baseline"]["Latex"],
                            (FHs["TQ"]["Latex"] + " > " + RMSthreshold_choice)],
                            colours=[FHs["Baseline"]["Colours"],
                                     "black"])
    fig_1.savefig("%s/Z0BDTResidual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0BDTResidual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0BDTResidual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0BDTResidual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual(MCs['Vertex']["array"],
                            [FHs["Baseline"]["array"], 
                            FHs["Chi2"]["array"]],
                            [FHs["Baseline"]["Latex"],
                            FHs["Chi2"]["Latex"]],
                            colours=[FHs["Baseline"]["Colours"],
                                             "black"])
    fig_1.savefig("%s/Z0Chi2Residual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0Chi2Residual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0Chi2Residual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual(MCs['Vertex']["array"],
                            [FHs["Baseline"]["array"],
                             MVA_bestRMS_array,
                             FHs["Chi2"]["array"]],
                            [FHs["Baseline"]["Latex"],
                             FHs["TQ"]["Latex"] + " > " + RMSthreshold_choice,
                             FHs["Chi2"]["Latex"]],
                            colours=[FHs["Baseline"]["Colours"],
                                     FHs["TQ"]["Colours"],
                                     "black"])
    fig_1.savefig("%s/Z0BDTChi2Residual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0BDTChi2Residual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0BDTChi2Residual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0BDTChi2Residual_NonLog.png" % outputFolder)


    plt.close()
    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual(MCs['Vertex']["array"],
                            [FHs["Baseline"]["array"], 
                            FHs["Combined"]["array"]],
                            [FHs["Baseline"]["Latex"],
                            FHs["Combined"]["Latex"]],
                            colours=[FHs["Baseline"]["Colours"],
                            "black"])
    fig_1.savefig("%s/Z0CombinedResidual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0CombinedResidual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0CombinedResidual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0CombinedResidual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(MCs['Vertex']["array"], (MCs['Vertex']["array"]-FHs["Baseline"]["array"]), bins=60,range=((-1*constants.max_z0,constants.max_z0),(-2*constants.max_z0,2*constants.max_z0)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("PV", horizontalalignment='right', x=1.0)
    ax.set_ylabel("PV - $z_0^{FH}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHerr_vs_z0.pdf" %  outputFolder)
    plt.savefig("%s/FHerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(MCs['Vertex']["array"], (FHs["Baseline"]["array"]), bins=60,range=((-1*constants.max_z0,constants.max_z0),(-1*constants.max_z0,constants.max_z0)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("PV", horizontalalignment='right', x=1.0)
    ax.set_ylabel("$z_0^{FH}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FH_vs_z0.pdf" %  outputFolder)
    plt.savefig("%s/FH_vs_z0.png" %  outputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    ax.plot(thresholds,predictedZ0_MVA_RMS_array,label="BDT Selection FH",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color='green')
    ax.plot(thresholds,np.full(len(thresholds),FHwidths[0]),label="Baseline FH",linestyle=linestyles[1],linewidth=LINEWIDTH,color='red')
    ax.plot(thresholds,np.full(len(thresholds),FHresWidths[0]),label="$\\eta$ Weighting FH",linestyle=linestyles[2],linewidth=LINEWIDTH,color=colours[2])
    #ax.plot(thresholds,np.full(len(thresholds),FHCombWidths[0]),label="Combined Function FH",linestyle=linestyles[3],linewidth=LINEWIDTH,color=colours[3])
    ax.plot(thresholds,np.full(len(thresholds),FHPVWidths[0]),label="No Fakes, No PU FH",linestyle=linestyles[4],linewidth=LINEWIDTH,color=colours[4])
    ax.plot(thresholds,np.full(len(thresholds),FHchi2Widths[0]),label="$\\chi^{2}$ Selection FH",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])

    ax.set_ylabel("$z_{0}^{PV}$ Residual RMS [cm]", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=1)
    plt.tight_layout()
    plt.savefig("%s/BDTRMSvsThreshold.pdf" %  outputFolder)
    plt.savefig("%s/BDTRMSvsThreshold.png" %  outputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    ax.plot(thresholds,predictedZ0_MVA_Quartile_array,label="BDT Selection FH",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color='green')
    ax.plot(thresholds,np.full(len(thresholds),FHwidths[1]),label="Baseline FH",linestyle=linestyles[1],linewidth=LINEWIDTH,color='red')
    ax.plot(thresholds,np.full(len(thresholds),FHresWidths[1]),label="$\\eta$ Weighting FH",linestyle=linestyles[2],linewidth=LINEWIDTH,color=colours[2])
    #ax.plot(thresholds,np.full(len(thresholds),FHCombWidths[1]),label="Combined Function FH",linestyle=linestyles[3],linewidth=LINEWIDTH,color=colours[3])
    ax.plot(thresholds,np.full(len(thresholds),FHPVWidths[1]),label="No Fakes, No PU FH",linestyle=linestyles[4],linewidth=LINEWIDTH,color=colours[4])
    ax.plot(thresholds,np.full(len(thresholds),FHchi2Widths[1]),label="$\\chi^{2}$ Selection FH",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])

    ax.set_ylabel("$z_{0}^{PV}$ Residual Quartile Width [cm]", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=3)
    plt.tight_layout()
    plt.savefig("%s/BDTQuartilevsThreshold.pdf" %  outputFolder)
    plt.savefig("%s/BDTQuartilevsThreshold.png" %  outputFolder)
    plt.close()

    # plt.close()
    # plt.clf()
    # fig,ax = plt.subplots(1,1,figsize=(10,10))
    # hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    # ax.plot(thresholds,predictedZ0_MVA_Centre_array,label="BDT FH",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color=colours[0])
    # ax.plot(thresholds,np.full(len(thresholds),FHwidths[2]),label="Base FH",linestyle=linestyles[1],linewidth=LINEWIDTH,color=colours[1])
    # ax.plot(thresholds,np.full(len(thresholds),FHresWidths[2]),label="$\\eta$ Corrected FH",linestyle=linestyles[2],linewidth=LINEWIDTH,color=colours[2])
    # ax.plot(thresholds,np.full(len(thresholds),FHCombWidths[2]),label="Combined Function FH",linestyle=linestyles[3],linewidth=LINEWIDTH,color=colours[3])
    # ax.plot(thresholds,np.full(len(thresholds),FHPVWidths[2]),label="Only PV FH",linestyle=linestyles[4],linewidth=LINEWIDTH,color=colours[4])
    # ax.plot(thresholds,np.full(len(thresholds),FHchi2Widths[2]),label="$\\chi^{2}$ Corrected FH",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])

    # ax.set_ylabel("$z_{0}^{PV}$ Residual Centre [cm]", horizontalalignment='right', y=1.0)
    # ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    # ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=3)
    # plt.tight_layout()
    # plt.savefig("%s/BDTCentrevsThreshold.pdf" %  outputFolder)
    # plt.savefig("%s/BDTCentrevsThreshold.png" %  outputFolder)
    # plt.close()

    # plt.close()
    # plt.clf()
    # fig,ax = plt.subplots(1,1,figsize=(10,10))
    # hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    # ax.plot(thresholds,predictedZ0_MVA_Efficiency_array,label="BDT FH",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color=colours[0])
    # ax.plot(thresholds,np.full(len(thresholds),FHwidths[3]),label="Base FH",linestyle=linestyles[1],linewidth=LINEWIDTH,color=colours[1])
    # ax.plot(thresholds,np.full(len(thresholds),FHresWidths[3]),label="$\\eta$ Corrected FH",linestyle=linestyles[2],linewidth=LINEWIDTH,color=colours[2])
    # ax.plot(thresholds,np.full(len(thresholds),FHCombWidths[3]),label="Combined Function FH",linestyle=linestyles[3],linewidth=LINEWIDTH,color=colours[3])
    # ax.plot(thresholds,np.full(len(thresholds),FHPVWidths[3]),label="Only PV FH",linestyle=linestyles[4],linewidth=LINEWIDTH,color=colours[4])
    # ax.plot(thresholds,np.full(len(thresholds),FHchi2Widths[3]),label="$\\chi^{2}$ Corrected FH",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])

    # ax.set_ylabel("Vertex Finding Efficiency (threshold 0.5 cm)", horizontalalignment='right', y=1.0)
    # ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    # ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=3)
    # plt.tight_layout()
    # plt.savefig("%s/BDTEfficiencyvsThreshold.pdf" %  outputFolder)
    # plt.savefig("%s/BDTEfficiencyvsThreshold.png" %  outputFolder)
    # plt.close()


############################## TRACK MET ####################################

predictedMET_MVA_array = {key: value for key, value in zip(thresholds, [np.zeros(1) for i in range(0,num_threshold)])}

predictedMET_MVA_RMS_array = np.zeros([num_threshold])
predictedMET_MVA_Quartile_array = np.zeros([num_threshold])
predictedMET_MVA_Centre_array = np.zeros([num_threshold])

MVAMET_histos = []
MVAMET_log_histos = []

for i in range(0,num_threshold):
    MET_MVA_array  = predictedMET_MVA[str(i/num_threshold)]
    predictedMET_MVA_array[str(i/num_threshold)] = MET_MVA_array
    Diff = MCs['MET']["array"] - MET_MVA_array

    Diff = Diff[~np.isnan(Diff)]

    predictedMET_MVA_RMS_array[i] = np.sqrt(np.mean(Diff**2))
    qMVA = np.percentile(Diff,[32,50,68])

    predictedMET_MVA_Quartile_array[i] = qMVA[2] - qMVA[0]
    predictedMET_MVA_Centre_array[i] = qMVA[1]

    hist,bin_edges = np.histogram((MCs['MET']["array"] - MET_MVA_array),bins=50,range=(-100,100))
    hist_log,bin_edges = np.histogram((MCs['MET']["array"] - MET_MVA_array),bins=50,range=(-300,300))
    MVAMET_histos.append(hist)
    MVAMET_log_histos.append(hist_log)

Quartilethreshold_choice = "0.9"#str(np.argmin(predictedMET_MVA_Quartile_array)/num_threshold)
RMSthreshold_choice= "0.9"#str(np.argmin(predictedMET_MVA_RMS_array)/num_threshold)

MVAMET_bestQ_array = predictedMET_MVA_array[Quartilethreshold_choice]
MVAMET_bestRMS_array = predictedMET_MVA_array[RMSthreshold_choice]

METBaselinewidths = calc_widths(MCs['MET']["array"],(MET["Baseline"]["array"]))
METOnlyz0Widths = calc_widths(MCs['MET']["array"],(MET["OnlyZ0"]["array"]))
METOnlyPVWidths = calc_widths(MCs['MET']["array"],(MET["OnlyPVs"]["array"]))
METChi2Widths = calc_widths(MCs['MET']["array"],(MET["Chi2"]["array"]))


############################## TRACK Relative MET ####################################

predictedRelativeMET_MVA_array = {key: value for key, value in zip(thresholds, [np.zeros(1) for i in range(0,num_threshold)])}

predictedRelativeMET_MVA_RMS_array = np.zeros([num_threshold])
predictedRelativeMET_MVA_Quartile_array = np.zeros([num_threshold])
predictedRelativeMET_MVA_Centre_array = np.zeros([num_threshold])

MVARelativeMET_histos = []
MVARelativeMET_log_histos = []

for i in range(0,num_threshold):
    RelativeMET_MVA_array  = predictedMET_MVA[str(i/num_threshold)]
    predictedRelativeMET_MVA_array[str(i/num_threshold)] = RelativeMET_MVA_array
    Diff = (RelativeMET_MVA_array-MCs['MET']["array"])/MCs['MET']["array"]
    Diff = Diff[~np.isnan(Diff)]

    predictedRelativeMET_MVA_RMS_array[i] = np.sqrt(np.mean(Diff**2))
    #rmse = metrics.mean_squared_error(y_actual,[0 for _ in y_actual], squared=False)
    qMVA = np.percentile(Diff,[32,50,68])

    predictedRelativeMET_MVA_Quartile_array[i] = qMVA[2] - qMVA[0]
    predictedRelativeMET_MVA_Centre_array[i] = qMVA[1]

    hist,bin_edges = np.histogram((RelativeMET_MVA_array-MCs['MET']["array"])/MCs['MET']["array"],bins=50,range=(-1,4))
    hist_log,bin_edges = np.histogram((RelativeMET_MVA_array-MCs['MET']["array"])/MCs['MET']["array"],bins=50,range=(-1,15))
    MVARelativeMET_histos.append(hist)
    MVARelativeMET_log_histos.append(hist_log)

Quartilethreshold_choice = "0.9"#str(np.argmin(predictedRelativeMET_MVA_Quartile_array)/num_threshold)
RMSthreshold_choice= "0.9"#str(np.argmin(predictedRelativeMET_MVA_RMS_array)/num_threshold)

print((predictedRelativeMET_MVA_Quartile_array))
print((predictedRelativeMET_MVA_RMS_array))
print(Quartilethreshold_choice)
print(RMSthreshold_choice)

MVARelativeMET_bestQ_array = predictedRelativeMET_MVA_array[Quartilethreshold_choice]
MVARelativeMET_bestRMS_array = predictedRelativeMET_MVA_array[RMSthreshold_choice]

RelativeMETBaselinewidths = calc_widths(MCs['MET']["array"],(MET["Baseline"]["array"]),relative=True)
RelativeMETOnlyz0Widths = calc_widths(MCs['MET']["array"],(MET["OnlyZ0"]["array"]),relative=True)
RelativeMETOnlyPVWidths = calc_widths(MCs['MET']["array"],(MET["OnlyPVs"]["array"]),relative=True)
RelativeMETChi2Widths = calc_widths(MCs['MET']["array"],(MET["Chi2"]["array"]),relative=True)

plot_MET = True
if plot_MET:
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(24,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    twod_hist = np.stack(MVAMET_histos, axis=1)
    twod_log_hist = np.stack(MVAMET_log_histos, axis=1)

    hist2d = ax[0].imshow(twod_hist,cmap=colormap,aspect='auto',extent=[0,1,-100,100])
    hist2d_log = ax[1].imshow(twod_log_hist,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),aspect='auto',cmap=colormap,extent=[0,1,-300,300])

    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_xlabel('MVA Threshold',ha="right",x=1)
    ax[0].set_ylabel('$E_{T,Track}^{miss}$ Residual [GeV]',ha="right",y=1)
    ax[1].set_xlabel('MVA Threshold',ha="right",x=1)
    ax[1].set_ylabel('$E_{T,Track}^{miss}$ Residual [GeV]',ha="right",y=1)

    cbar = plt.colorbar(hist2d , ax=ax[0])
    cbar.set_label('# Events')

    cbar = plt.colorbar(hist2d_log , ax=ax[1])
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/MET2dhist.pdf" % METoutputFolder)
    plt.savefig("%s/MET2dhist.png" % METoutputFolder)


    ### Relative ### 

    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(24,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    twod_hist = np.stack(MVARelativeMET_histos, axis=1)
    twod_log_hist = np.stack(MVARelativeMET_log_histos, axis=1)

    hist2d = ax[0].imshow(twod_hist,cmap=colormap,aspect='auto',extent=[0,1,-1,4])
    hist2d_log = ax[1].imshow(twod_log_hist,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),aspect='auto',cmap=colormap,extent=[0,1,-1,15])

    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_xlabel('MVA Threshold',ha="right",x=1)
    ax[0].set_ylabel('$E_{T,Track}^{miss}$ Resolution [GeV]',ha="right",y=1)
    ax[1].set_xlabel('MVA Threshold',ha="right",x=1)
    ax[1].set_ylabel('$E_{T,Track}^{miss}$ Resolution [GeV]',ha="right",y=1)

    cbar = plt.colorbar(hist2d , ax=ax[0])
    cbar.set_label('# Events')

    cbar = plt.colorbar(hist2d_log , ax=ax[1])
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/RelativeMET2dhist.pdf" % METoutputFolder)
    plt.savefig("%s/RelativeMET2dhist.png" % METoutputFolder)


    plt.close()
    plt.clf()
    figure=plotMET_residual(MCs['MET']["array"],
                           [MVAMET_bestRMS_array,
                            #MET["Baseline"]["array"], 
                            MET["OnlyZ0"]["array"],
                            #MET["OnlyPVs"]["array"],
                            MET["Chi2"]["array"]],
                            [MET["TQ"]["Latex"],
                            #MET["Baseline"]["Latex"],
                            MET["OnlyZ0"]["Latex"],
                            #MET["OnlyPVs"]["Latex"],
                            MET["Chi2"]["Latex"]],
                            colours=[MET["TQ"]["Colours"],
                            #MET["Baseline"]["Colours"],
                            MET["OnlyZ0"]["Colours"],
                            #MET["OnlyPVs"]["Colours"],
                            MET["Chi2"]["Colours"]])
    plt.savefig("%s/METResidual.pdf" % METoutputFolder)
    plt.savefig("%s/METResidual.png" % METoutputFolder)

    plt.close()
    plt.clf()
    figure=plotMET_residual(MCs['MET']["array"],
                           [MVARelativeMET_bestRMS_array,
                            #MET["Baseline"]["array"], 
                            MET["OnlyZ0"]["array"],
                            #MET["OnlyPVs"]["array"],
                            MET["Chi2"]["array"]],
                            [MET["TQ"]["Latex"],
                            #MET["Baseline"]["Latex"],
                            MET["OnlyZ0"]["Latex"],
                            #MET["OnlyPVs"]["Latex"],
                            MET["Chi2"]["Latex"]],
                            colours=[MET["TQ"]["Colours"],
                            #MET["Baseline"]["Colours"],
                            MET["OnlyZ0"]["Colours"],
                            #MET["OnlyPVs"]["Colours"],
                            MET["Chi2"]["Colours"]],
                            relative=True)
    plt.savefig("%s/METResolution.pdf" % METoutputFolder)
    plt.savefig("%s/METResolution.png" % METoutputFolder)

    ######################################################

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(MCs['MET']["array"], (MCs['MET']["array"]-MET["Baseline"]["array"]), bins=60,range=((0,300),(-100,100)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("$E_{T,True}^{miss}$", horizontalalignment='right', x=1.0)
    ax.set_ylabel("$E_{T,True}^{miss} - E_{T,Tracks,Baseline}^{miss}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/METerr_vs_MET.pdf" %  METoutputFolder)
    plt.savefig("%s/METerr_vs_MET.png" %  METoutputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(MCs['MET']["array"], (MET["Baseline"]["array"]), bins=60,range=((0,300),(0,300)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("$E_{T,True}^{miss}$", horizontalalignment='right', x=1.0)
    ax.set_ylabel("$E_{T,Tracks}^{miss}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/MET_vs_MET.pdf" %  METoutputFolder)
    plt.savefig("%s/MET_vs_MET.png" %  METoutputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

    ax.plot(thresholds,predictedMET_MVA_RMS_array,label="Reco $E_{T,Tracks}^{miss}$ BDT Cuts",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color="g")
    #ax.plot(thresholds,np.full(len(thresholds),METBaselinewidths[0]),label="Reco $E_{T,Tracks}^{miss}$",linestyle=linestyles[1],linewidth=LINEWIDTH,color="r")
    ax.plot(thresholds,np.full(len(thresholds),METOnlyz0Widths[0]),label="Reco $E_{T,Tracks}^{miss}$",linestyle=linestyles[2],linewidth=LINEWIDTH,color="pink")
    ax.plot(thresholds,np.full(len(thresholds),METOnlyPVWidths[0]),label="True $E_{T,Tracks}^{miss}$ ",linestyle=linestyles[3],linewidth=LINEWIDTH,color="k")
    ax.plot(thresholds,np.full(len(thresholds),METChi2Widths[0]),label="Reco $E_{T,Tracks}^{miss}$ $\\chi^2$ Cuts",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])

    ax.set_ylabel("$E_{T,Tracks}^{miss}$ Residual RMS [GeV]", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    #ax.set_yscale("log")
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=7)
    plt.tight_layout()
    plt.savefig("%s/BDTRMSMETvsThreshold.pdf" %  METoutputFolder)
    plt.savefig("%s/BDTRMSMETvsThreshold.png" %  METoutputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    ax.plot(thresholds,predictedMET_MVA_Quartile_array,label="Reco $E_{T,Tracks}^{miss}$ MVA Cuts",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color="g")
    #ax.plot(thresholds,np.full(len(thresholds),METBaselinewidths[1]),label="Reco $E_{T,Tracks}^{miss}$",linestyle=linestyles[1],linewidth=LINEWIDTH,color="r")
    ax.plot(thresholds,np.full(len(thresholds),METOnlyz0Widths[1]),label="Reco $E_{T,Tracks}^{miss}$",linestyle=linestyles[2],linewidth=LINEWIDTH,color="pink")
    ax.plot(thresholds,np.full(len(thresholds),METOnlyPVWidths[1]),label="True $E_{T,Tracks}^{miss}$ ",linestyle=linestyles[3],linewidth=LINEWIDTH,color="k")
    ax.plot(thresholds,np.full(len(thresholds),METChi2Widths[1]),label="Reco $E_{T,Tracks}^{miss}$ $\\chi^2$ Cuts",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])

    ax.set_ylabel("$E_{T,Tracks}^{miss}$ Residual Quartile Width [GeV]", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=1)
    #ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("%s/BDTMETQuartilevsThreshold.pdf" %  METoutputFolder)
    plt.savefig("%s/BDTMETQuartilevsThreshold.png" %  METoutputFolder)
    plt.close()

    ###### Relative #######

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

    ax.plot(thresholds,predictedRelativeMET_MVA_RMS_array,label="Reco $E_{T,Tracks}^{miss}$ MVA Cuts",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color="g")
    ax.plot(thresholds,np.full(len(thresholds),RelativeMETOnlyz0Widths[0]),label="Reco $E_{T,Tracks}^{miss}$",linestyle=linestyles[2],linewidth=LINEWIDTH,color="pink")
    ax.plot(thresholds,np.full(len(thresholds),RelativeMETOnlyPVWidths[0]),label="True $E_{T,Tracks}^{miss}$ ",linestyle=linestyles[3],linewidth=LINEWIDTH,color="k")
    ax.plot(thresholds,np.full(len(thresholds),RelativeMETChi2Widths[0]),label="Reco $E_{T,Tracks}^{miss}$ $\\chi^2$ Cuts",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])

    ax.set_ylabel("$E_{T,Tracks}^{miss}$ Resolution RMS", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    #ax.set_yscale("log")
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=7)
    plt.tight_layout()
    plt.savefig("%s/BDTRMSRelativeMETvsThreshold.pdf" %  METoutputFolder)
    plt.savefig("%s/BDTRMSRelativeMETvsThreshold.png" %  METoutputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    ax.plot(thresholds,predictedRelativeMET_MVA_Quartile_array,label="Reco $E_{T,Tracks}^{miss}$ MVA Cuts",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color="g")
    ax.plot(thresholds,np.full(len(thresholds),RelativeMETOnlyz0Widths[1]),label="Reco $E_{T,Tracks}^{miss}$",linestyle=linestyles[2],linewidth=LINEWIDTH,color="pink")
    ax.plot(thresholds,np.full(len(thresholds),RelativeMETOnlyPVWidths[1]),label="True $E_{T,Tracks}^{miss}$ ",linestyle=linestyles[3],linewidth=LINEWIDTH,color="k")
    ax.plot(thresholds,np.full(len(thresholds),RelativeMETChi2Widths[1]),label="Reco $E_{T,Tracks}^{miss}$ $\\chi^2$ Cuts",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])

    ax.set_ylabel("$E_{T,Tracks}^{miss}$ Resolution Quartile Width", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=1)
    #ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("%s/BDTRelativeMETQuartilevsThreshold.pdf" %  METoutputFolder)
    plt.savefig("%s/BDTRelativeMETQuartilevsThreshold.png" %  METoutputFolder)
    plt.close()

    plt.close()
    plt.clf()
    figure=plotMET_residual(MCs['MET']["array"],
                           [MVARelativeMET_bestRMS_array,
                            #MET["Baseline"]["array"], 
                            MET["OnlyZ0"]["array"],
                            #MET["OnlyPVs"]["array"],
                            MET["Chi2"]["array"]],
                            [MET["TQ"]["Latex"],
                            #MET["Baseline"]["Latex"],
                            MET["OnlyZ0"]["Latex"],
                            #MET["OnlyPVs"]["Latex"],
                            MET["Chi2"]["Latex"]],
                            colours=[MET["TQ"]["Colours"],
                            #MET["Baseline"]["Colours"],
                            MET["OnlyZ0"]["Colours"],
                            #MET["OnlyPVs"]["Colours"],
                            MET["Chi2"]["Colours"]],
                            relative=True, logbins=True)
    plt.savefig("%s/METResolutionLogBin.pdf" % METoutputFolder)
    plt.savefig("%s/METResolutionLogBin.png" % METoutputFolder)

    ######################################################

    chi2_eff = 190
    baseline_eff = 5000 
    mva_effs =  []#[, ,   ,   ,   ,  , ,  5000  ,5000,]


    chi2_rate = 63
    baseline_rate = 186
    mva_rate = []#[   , ,   ,   ,   ,   ,   , 87 , 81, 75]
                #0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
    
    #noselection_eff = 



    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

    ax.plot(mva_rate,mva_effs,label="Reco $E_{T,Tracks}^{miss}$ MVA Cuts",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color="g")
    ax.plot(chi2_rate,chi2_eff,label="Reco $E_{T,Tracks}^{miss}$ $\\chi^2$ Cuts$",linestyle=linestyles[2],color="blue",markersize=10,marker='o')
    ax.plot(baseline_rate,baseline_eff,label="Reco $E_{T,Tracks}^{miss}$",linestyle=linestyles[2],color="red",markersize=10,marker='o')

    ax.set_ylabel("$E_{T,Tracks}^{miss}$ Threshold of 95% Efficiency [GeV]", horizontalalignment='right', y=1.0)
    ax.set_xlabel("$E_{T,Tracks}^{miss}$ Threshold of 35 kHz rate [GeV]", horizontalalignment='right', x=1.0)
    #ax.set_yscale("log")
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=7)
    plt.tight_layout()
    plt.savefig("%s/Rate_Efficiency.pdf" %  METoutputFolder)
    plt.savefig("%s/Rate_Efficiency.png" %  METoutputFolder)
    plt.close()
