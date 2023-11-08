import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")

import sklearn.metrics as metrics
from textwrap import wrap


"""
These function define evaluation plots to be run on the output of the model

"""

# Setup plotting to CMS style
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
plt.rc('axes', linewidth=LINEWIDTH+2)              # thickness of axes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-2)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 20
matplotlib.rcParams['xtick.major.width'] = 5
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 4

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 5
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 4

colours=["red","black","blue","orange","purple","goldenrod",'green']
linestyles = ["-","--","dotted",(0, (3, 5, 1, 5)),(0, (3, 5, 1, 5, 1, 5)),(0, (3, 10, 1, 10)),(0, (3, 10, 1, 10, 1, 10))]


max_z0 = 20.46912512
nbins = 256

def plot_event(twod_histo,y,feature_names,max_z0,nbins):
    fig,ax = plt.subplots(1,1,figsize=(24,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    nan_array = np.zeros_like(twod_histo[0])
    nan_array[:] = np.NaN
    nan_array = np.expand_dims(nan_array,axis=0)
    twod_histo = np.vstack([twod_histo,nan_array])
    hist2d = ax.imshow(twod_histo,cmap=colormap,aspect='auto',extent=[-1*max_z0,max_z0,0,len(feature_names)])

    ax.grid(True,axis='y',linewidth=2)
    ax.grid(True,axis='x',linewidth=1)
    ax.set_ylabel('Track Feature',ha="right",y=1)
    ax.set_xlabel('Track $z_{0}$ [cm]',ha="right",x=1)
        
    ax.set_yticklabels(feature_names)
    ax.set_yticks(np.array([1,2,3,4,5,6,7,8,9,10,11]))

    #rect = plt.Rectangle((y-((2.5*max_z0)/nbins), 1), 5*max_z0/nbins, len(feature_names),
    #                                fill=False,linewidth=2,linestyle='--',edgecolor='r')
    #ax.add_patch(rect)
    ax.text(0, 0.5, "Sim Score " + str(y), color='k')

    cbar = plt.colorbar(hist2d , ax=ax)
    cbar.set_label('Weighted Density')

    cbar.set_label('Weighted Density')
    ax.tick_params(axis='y', which='minor', right=False,left=False)
    plt.tight_layout()
    return fig

def plotz0_residual(actual,predicted,names,title="None",max_z0=20.46912512,colours=colours,linestyles=linestyles):
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    
    items = 0
    for i,prediction in enumerate(predicted):
        FH = (actual - prediction)
        qz0_FH = np.percentile(FH,[32,50,68])
        ax[0].hist(FH,bins=50,range=(-1*max_z0,max_z0),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nRMS = %.4f" 
                 %(names[i],np.sqrt(np.mean(FH**2))),LEGEND_WIDTH)),density=True)
        ax[1].hist(FH,bins=50,range=(-1,1),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                 %(names[i],qz0_FH[2]-qz0_FH[0]),LEGEND_WIDTH)),density=True)
        items+=1
    
    ax[0].grid(True)
    ax[0].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[0].set_ylabel('Events',ha="right",y=1)
    ax[0].set_yscale("log")
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    ax[1].grid(True)
    ax[1].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[1].set_ylabel('Events',ha="right",y=1)
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_split_z0_residual(actual,predicted,names,title="None",max_z0=20.46912512,colours=colours):
    fig_1,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    items = 0

    for i,prediction in enumerate(predicted):
        FH = (actual - prediction)
        qz0_FH = np.percentile(FH,[32,50,68])
        ax.hist(FH,bins=50,range=(-1*max_z0,max_z0),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items], linestyle=linestyles[items])
        ax.plot(0,1,label='\n'.join(wrap(f"%s \n RMS = %.4f" 
                 %(names[i],np.sqrt(np.mean(FH**2))),LEGEND_WIDTH)),color = colours[items],markersize=0,linewidth=LINEWIDTH)
        items+=1

    ax.grid(True)
    ax.set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax.set_ylabel('Events',ha="right",y=1)
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),frameon=True,facecolor='w',edgecolor='w')
    ax.set_yscale("log")
    #ax.set_ylim([5,200000])
    fig_1.tight_layout()

    #============================================================================================#
    plt.close()
    plt.clf()
    fig_2,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    items = 0

    for i,prediction in enumerate(predicted):
        FH = (actual - prediction)
        qz0_FH = np.percentile(FH,[32,50,68])
        ax.hist(FH,bins=50,range=(-1,1),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items], linestyle=linestyles[items])
        ax.plot(0,1,label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                 %(names[i],qz0_FH[2]-qz0_FH[0]),LEGEND_WIDTH)),color = colours[items],markersize=0,linewidth=LINEWIDTH)
        items+=1

    ax.grid(True)
    ax.set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax.set_ylabel('Events',ha="right",y=1)
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),frameon=True,facecolor='w',edgecolor='w')
    #ax.set_ylim([0,65000])
    fig_2.tight_layout()

    return fig_1,fig_2

def plot_histo(histos,name,title,range=(0,1)):
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])

    items = 0
    for i,histo in enumerate(histos):

        ax[0].hist(histo,bins=50,range=range,histtype="step",
                    linewidth=LINEWIDTH,
                    color = colours[items],
                    label=name[i],
                    density=True)
        ax[1].hist(histo,bins=50,range=range,histtype="step",
                    linewidth=LINEWIDTH,
                    color = colours[items],
                    label=name[i],
                    density=True)
        items+=1

    
    ax[0].grid(True)
    ax[0].set_xlabel("Score",ha="right",x=1)
    ax[0].set_ylabel('Events',ha="right",y=1)
    ax[0].set_yscale("log")
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    ax[1].grid(True)
    ax[1].set_xlabel("Score",ha="right",x=1)
    ax[1].set_ylabel('Events',ha="right",y=1)
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    plt.suptitle(title)
    plt.tight_layout()
    return fig

def calculate_rates(actual,prediction,Nthresholds=1000):
    thresholds = np.linspace(0,1,Nthresholds)
    precision = []
    recall = []
    FPR= []

    auc = metrics.roc_auc_score(actual,prediction)

    for j,threshold in enumerate(thresholds):
            # Find number of true negatives, false positive, false negatives and true positives when decision bounary == threshold
            #print("Threshold: ",j,"out of ",Nthresholds)
            tn, fp, fn, tp = metrics.confusion_matrix(actual, prediction>threshold).ravel()
            precision.append( tp / (tp + fp) )
            recall.append(tp / (tp + fn) )
            FPR.append(fp / (fp + tn) )

    return [precision,recall,FPR,auc]

def plotPV_roc(rates,names,title="None",colours=colours):
    '''
    Plots reciever operating characteristic curve for output of a predicition model

    Takes: 
        actual: a numpy array of true values 0 or 1
        predictions: a list of numpy arrays, each array same length as actual which are probabilities of coming from class 1, float between 0 and 1
        names: a list of strings naming each of the prediciton arrays
        Nthresholds: how many thresholds between 0 and 1 to calcuate the TPR, FPR etc.
        colours: list of matplotlib colours to be used for each item in the predictions list

    '''
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    #hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])

    items = 0
    # Iterate through predictions
    for i,rate in enumerate(rates):
        precision = rate[0]
        recall = rate[1]
        FPR= rate[2]

        # Plot precision recall and ROC curves
        #ax[0].plot(recall,precision,label=str(names[i]),linewidth=LINEWIDTH,color=colours[items])
        ax.plot(recall,FPR,linewidth=LINEWIDTH,label='\n'.join(wrap(f"%s AUC: %.4f" %(names[i],rate[3]),LEGEND_WIDTH)),color=colours[items])
        items += 1

    # ax[0].grid(True)
    # ax[0].set_xlabel('Efficiency',ha="right",x=1)
    # ax[0].set_ylabel('Purity',ha="right",y=1)
    # ax[0].set_xlim([0,1.0])
    # ax[0].set_ylim([0,1.0])
    # ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    ax.grid(True)
    #ax.set_yscale("log")
    ax.set_xlabel('Vertex ID True Positive Rate',ha="right",x=1)
    ax.set_ylabel('Vertex ID False Positive Rate',ha="right",y=1)
    ax.set_xlim([0.0,1])
    ax.set_ylim([0.0,1.0])
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_split_histo(actual,variable,variable_name,range=(0,1),bins=100,title="None"):
    pv_track_sel = actual == 1
    pu_track_sel = actual == 0
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(variable[pv_track_sel],range=range, bins=bins, label="Real Vertices", density=True,histtype="stepfilled",color='g',alpha=0.7,linewidth=LINEWIDTH)
    ax.hist(variable[pu_track_sel],range=range, bins=bins, label="Fake Vertices", density=True,histtype="stepfilled",color='r',alpha=0.7,linewidth=LINEWIDTH)
    ax.set_xlabel(variable_name, horizontalalignment='right', x=1.0)
    ax.set_ylabel("Fraction of Events", horizontalalignment='right', y=1.0)
    #ax.set_yscale("log")
    ax.legend()
    ax.tick_params(axis='x', which='minor', bottom=False,top=False)
    plt.suptitle(title)
    plt.tight_layout()

    return fig

def plotMET_residual(actual,predictions,names,colours=colours,relative=False,logbins=False):
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    if relative:
        range = (-1,4)
        logrange = (-1,15) 
    else:
        range = (-100,100)
        logrange = (-300,300) 
        

    if logbins:
        bins = [0,1,2,3,4.5,6,8,10,15]#[0,5,10,20,30,45,60,80,100]
    else:
        bins = np.linspace(logrange[0],logrange[1],50)

    
    
    items = 0
    for i,prediction in enumerate(predictions):
        if relative:
            FH = (prediction - actual) / actual
            temp_actual = actual[~np.isnan(FH)]
            FH = FH[~np.isnan(FH)]
            temp_actual = temp_actual[np.isfinite(FH)]
            FH = FH[np.isfinite(FH)]

            if logbins:
                FH = FH + 1  
        else:
            FH = (prediction - actual)
            temp_actual = actual
        qz0_FH = np.percentile(FH,[32,50,68])

        ax[0].hist(FH,bins=bins,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s RMS = %.4f" 
                 %(names[i],np.sqrt(np.mean(FH**2))))),density=True)
        ax[1].hist(FH,bins=50,range=range,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s Quartile Width = %.4f   Centre = %.4f" 
                 %(names[i],qz0_FH[2]-qz0_FH[0], qz0_FH[1]),25)),density=True)
        items+=1
    
    ax[0].grid(True)
    
    ax[0].set_ylabel('Events',ha="right",y=1)
    ax[0].set_yscale("log")
    ax[0].legend(loc='upper right', bbox_to_anchor=(0.95, 0.95)) 

    ax[1].grid(True)
    
    ax[1].set_ylabel('Events',ha="right",y=1)
    ax[1].legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

    if relative:
        ax[0].set_xlabel('$E_{T,Tracks}^{miss}$ Resolution $\\frac{Reco - True}{True}$',ha="right",x=1)
        ax[1].set_xlabel('$E_{T,Tracks}^{miss}$ Resolution $\\frac{Reco - True}{True}$',ha="right",x=1)

    else:
        ax[0].set_xlabel('$E_{T}^{miss}$ Residual [GeV]',ha="right",x=1)
        ax[1].set_xlabel('$E_{T}^{miss}$ Residual [GeV]',ha="right",x=1)


    plt.tight_layout()
    return fig
