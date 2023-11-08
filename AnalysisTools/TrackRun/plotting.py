import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sklearn.metrics as metrics
from textwrap import wrap
import pandas as pd
import gc


"""
These function define evaluation plots to be run on the output of the model

"""

# Setup plotting to CMS style
hep.cms.label()
hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 35

LEGEND_WIDTH = 20
LINEWIDTH = 3
MARKERSIZE = 20

colormap = "jet"

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=LINEWIDTH+2)              # thickness of axes
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE-2)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 20
matplotlib.rcParams['xtick.major.width'] = 5
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 4

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 5
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 4

#colours=["green","red","blue","black","orange","purple","goldenrod"]
colours = ["black","red","orange","green", "blue","black","red","orange","green", "blue"]
linestyles = ["-","--","dotted",(0, (3, 5, 1, 5)),(0, (3, 5, 1,1,1,5,)),(0, (3, 10, 1, 10)),(0, (3, 10, 1, 10, 1, 10)),
              (0, (3, 10, 1, 10, 1, 10)),(0, (3, 10, 1, 10)),(0, (3, 5, 1,1,1,5,)),(0, (3, 5, 1, 5)),"dotted","--","-"]

def set_nlaymissinterior(hitpat):
    bin_hitpat = np.binary_repr(hitpat)
    bin_hitpat = bin_hitpat.strip('0')
    return bin_hitpat.count('0')

def tanl(eta):
    return 1*np.sign(eta)*abs(np.sinh(eta))

def save_branches(file,dataframe,feature_list,max_batch=10000,name=""):
    batch = 0
    TTTrackDF = pd.DataFrame()
    for array in file.iterate(library="numpy",filter_name=feature_list,step_size=1000):
        temp_array = pd.DataFrame()
        for feature in feature_list:
            temp_array[feature] = np.concatenate(array[feature]).ravel()
        TTTrackDF = pd.concat([TTTrackDF,temp_array],ignore_index=False)
        print("Cumulative", name, "read: ", len(TTTrackDF))
        del [temp_array]
        del [array]
        batch += 1
        if batch >= max_batch:
            break
    trackskept = feature_list

    Tracks = TTTrackDF[trackskept]
    Tracks.reset_index(inplace=True)
    Tracks.dropna(inplace=True)
    del [TTTrackDF]

    for j in trackskept:
        dataframe[j] = Tracks[j]
    del [Tracks]

    infs = np.where(np.asanyarray(np.isnan(dataframe)))[0]
    dataframe.drop(infs, inplace=True)
    print("Reading Complete, read: ", len(dataframe), name)

    gc.collect()

    return dataframe

def plot_variable(variable,variable_name,title,xrange=(0,1),bins=50,yrange=None,density=False,log=False):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    if bins == 0:
        bins = int(xrange[1]-xrange[0]) + 1
    ax.hist(variable,bins=bins,range=xrange,histtype="step",
                    linewidth=LINEWIDTH,
                    color = colours[0],
                    density=density)

    
    ax.grid(True)
    ax.set_xlabel(variable_name,ha="right",x=1)
    if log: ax.set_yscale("log")

    if density:
        ax.set_ylabel('# Tracks / # Total Tracks',ha="right",y=1)
    else:
        ax.set_ylabel('# Tracks',ha="right",y=1)

    if yrange != None:
        ax.set_ylim(yrange)

    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_split_variable(variable,variable_name,splits,split_names,title,xrange=(0,1),bins=50,yrange=None,density=False,log=False,lloc='upper right'):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    if bins == 0:
        bins = int(xrange[1]-xrange[0])
        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
        #for tick in ax.xaxis.get_major_ticks():
        #    tick.label1.set_horizontalalignment('center')

        dx = 55/72; dy = 0/72. 
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
        ax.xaxis.get_major_ticks()[0].label1.set_visible(False) ## set last x tick label invisible

    for i,name in enumerate(split_names):

        ax.hist(variable[splits[i]],bins=bins,range=xrange,histtype="step",
                        linewidth=LINEWIDTH,
                        color = colours[i],
                        label = name,
                        density=density)

    
    ax.grid(True)
    ax.set_xlabel(variable_name,ha="right",x=1)
    if density:
        ax.set_ylabel('# Tracks / # Total Tracks',ha="right",y=1)
    else:
        ax.set_ylabel('# Tracks',ha="right",y=1)
    ax.legend(loc=lloc, frameon=True,facecolor='w',edgecolor='w')
    if log: ax.set_yscale("log")

    if yrange != None:
        ax.set_ylim(yrange)

    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_2d(variable_one,variable_two,range_one,range_two,name_one,name_two,title):
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(variable_one, variable_two, range=(range_one,range_two), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel(name_one, horizontalalignment='right', x=1.0)
    ax.set_ylabel(name_two, horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    #ax.vlines(0,-20,20,linewidth=3,linestyle='dashed',color='k')
    plt.suptitle(title)
    plt.tight_layout()
    return fig
   
def plot_multiple_variable(variables,variable_name,labels,title,xrange=(0,1),yrange=None,density=False,log=False,lloc='upper right',bins=50):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

    for i,variable in enumerate(variables):

        ax.hist(variable,bins=bins,range=xrange,histtype="step",
                        linewidth=LINEWIDTH,
                        color = colours[i],
                        density=density,
                        label = labels[i],
                        linestyle = linestyles[i]
                    )
    
    ax.grid(True)
    ax.set_xlabel(variable_name,ha="right",x=1)
    if log: ax.set_yscale("log")
    if density:
        ax.set_ylabel('# Tracks / # Total Tracks',ha="right",y=1)
    else:
        ax.set_ylabel('# Tracks',ha="right",y=1)
    ax.legend(loc=lloc, frameon=True,facecolor='w',edgecolor='w')

    if yrange != None:
        ax.set_ylim(yrange)

    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_multiple_variable_ratio(variables,variable_name,labels,title,xrange=(0,1),yrange=None,density=False,bins=50,xratios=None,lloc='upper left',log=False,textpos=(0.05,0.9)):
    fig, ax = plt.subplots(figsize=(10,13))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    

    for i,variable in enumerate(variables):

        ax.hist(variable,bins=bins,range=xrange,histtype="step",
                        linewidth=LINEWIDTH,
                        color = colours[i],
                        density=density,
                        label = labels[i],
                        linestyle = linestyles[i]
                        )


    
    bins = np.linspace(xrange[0]+((xrange[1]-xrange[0])/(2*bins)),xrange[1]+((xrange[1]-xrange[0])/(2*bins)),bins+1)
   
    baseline = np.histogram(variables[0],bins=bins,range=xrange)
    #ax.set_aspect(1.)
    ax.xaxis.set_tick_params(labelbottom=False)
   # ax.text(textpos[0],textpos[1], "Tracks in t$\overline{t}$ + PU = 200\n $p_T$ > 2 GeV, $|\eta|$ < 2.4", color='k',fontsize=SMALL_SIZE+1,style='italic')

    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(ax)
    # below height and pad are in inches
    ax_ratio = divider.append_axes("bottom", 2.0, pad=0.5, sharex=ax)

    # make some labels invisible
    ax_ratio.xaxis.set_tick_params(labeltop=False)
    
    for i,variable in enumerate(variables):
        if i == 0:
            continue
        else:
            binned = np.histogram(variable,bins=bins,range=xrange)
            #bins = bins+1
            #ax_ratio.scatter(bins[:-1],binned[0]/baseline[0],color=colours[i],s=50)
            if xratios == None:
                xs = bins[:-1]
            else:
                xs = xratios
            ax_ratio.errorbar(xs,binned[0]/baseline[0],yerr=(1/np.sqrt(binned[0])),color=colours[i],markersize=7,marker='o',linewidth=0,elinewidth=3)
    ax_ratio.axhline(1,0,xrange[1],linestyle="--",color='k')

    if len(bins) == 5:
        print(bins)
        for tick in ax_ratio.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)

        dx = 55/72; dy = 0/72. 
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

        # apply offset transform to all x ticklabels.
        for label in ax_ratio.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
            print(label)
        ax_ratio.xaxis.get_major_ticks()[-2].label1.set_visible(False) ## set last x tick label invisible
        #ax_ratio.set_xticklabels(ax_ratio.xaxis.get_major_ticks()[:-1])
    #ax_ratio.grid(True)
    #ax_ratio.set_xlabel(variable_name,ha="right",x=1)
    ax.set_ylabel("# Tracks",ha="right",y=1)

    ax.legend(loc=lloc, frameon=True,facecolor='w',edgecolor='w')

    if log:
        ax.set_yscale("log")


    #ax_ratio.grid(True)
    ax_ratio.set_xlabel(variable_name,ha="right",x=1)
    ax_ratio.set_ylabel("Ratio",ha="left",y=0)
    ax_ratio.set_ylim(yrange)
    #ax_ratio.set_yscale("log")

    plt.suptitle(title)
    plt.tight_layout()
    return fig
