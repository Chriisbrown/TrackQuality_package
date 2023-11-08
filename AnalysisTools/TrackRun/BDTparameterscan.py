import pandas as pd
from plotting import *
DF = pd.read_csv("gamma_fixed_scan.txt",delimiter=" ")
print(DF)

#precisions = ['ap_fixed<12,6>','ap_fixed<12,5>','ap_fixed<11,6>','ap_fixed<11,5>','ap_fixed<10,6>','ap_fixed<10,5>','ap_fixed<9,6>','ap_fixed<9,5>','ap_fixed<8,6>','ap_fixed<8,5>']

#precisions = ['ap_fixed<32,16>','ap_fixed<12,6>','ap_fixed<12,5>','ap_fixed<11,6>','ap_fixed<11,5>','ap_fixed<10,5>','ap_fixed<10,4>','ap_fixed<9,5>','ap_fixed<9,4>','ap_fixed<8,5>','ap_fixed<8,4>']
precisions = ['ap_fixed<12,6>','ap_fixed<12,5>','ap_fixed<11,6>','ap_fixed<11,5>','ap_fixed<10,6>','ap_fixed<10,5>','ap_fixed<9,6>','ap_fixed<9,5>','ap_fixed<8,6>','ap_fixed<8,5>']

gammas = [0.0,1.0,10.0,20.0,30.0,50.0,100.0]

roc_percentage = np.zeros((len(precisions),len(gammas)))
LUT = np.zeros((len(precisions),len(gammas)))
FF = np.zeros((len(precisions),len(gammas)))

for i,precision in enumerate(precisions):
    tempDF = DF[DF['precision']==precision]
    for j,gamma in enumerate(gammas):
        tempDF2 = tempDF[tempDF['gamma']==gamma]
        try:
            roc_percentage[i,j] = tempDF2['hls_roc'].values[0] #/  tempDF2['ref_roc'].values[0]
            LUT[i,j] = tempDF2['LUT'].values[0]*100
            FF[i,j] = tempDF2['FF'].values[0]*100
        except:
            pass

def plot_im(image,axislabel,minval=0.1,maxval=1.0):
    fig,ax = plt.subplots(1,1,figsize=(18,16))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)  

    #im = ax.imshow(image,norm=matplotlib.colors.LogNorm(vmin=minval,vmax=maxval),cmap='viridis')
    im = ax.imshow(image,vmin=minval,vmax=maxval,cmap='viridis')


    ax.set_ylabel("Fixed-Point Precision", horizontalalignment='right', y=1.0)
    ax.set_yticks(np.arange(len(precisions))+0.5)
    ax.set_yticklabels(precisions)
    ax.set_xlabel("$\\gamma$ Pruning Parameter", horizontalalignment='right', x=1.0)
    ax.set_xticks(np.arange(len(gammas))+0.5)
    ax.set_xticklabels(gammas)

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)

    dx = -50/72; dy = 0/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

            # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)


    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)

    dx = 0/72; dy = 50/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    for label in ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)


    cbar = plt.colorbar(im,ax=ax)
    cbar.set_label(axislabel,labelpad=20)
    #plt.suptitle(title)
    plt.tight_layout()
    return fig


plt.clf()
fig = plot_im(roc_percentage,'Conifer ROC Score',minval=0.9608,maxval=DF['hls_roc'].max())
plt.savefig("ROCScore.png")

plt.clf()
fig = plot_im(LUT,'Percentage LUT Usage',minval=1.8,maxval=2.1)
plt.savefig("LUT.png")

plt.clf()
fig = plot_im(FF,'Percentage FF Usage',minval=1.18,maxval=1.3)
plt.savefig("FF.png")
