from Models.GBDTTrackQualityModel import XGBoostClassifierModel
from Datasets.Dataset import *
from Utils.util import *
from Utils.TreeVis import *
import numpy as np
from scipy.special import expit
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
model = XGBoostClassifierModel("Degredation_0")
model.load_model("/home/ryanm124/ml_hackathon/TrackQuality_package/Scripts/Projects/Retrained/Models/","Retrained0123_XGB")
treedict = convert(model.model.get_booster())
gains = []
treeNum = []
cutValues = [2000,1000,400,0]
gainSum = np.zeros_like(cutValues)
percentGain = []
j=0
for i in range(treedict['n_trees']):
    print("Tree #: ",i)
    print(treedict['trees'][i][0]['value'][0])
    gains.append(treedict['trees'][i][0]['value'][0])
    treeNum.append(i)
    if(treedict['trees'][i][0]['value'][0]<cutValues[j]):
        j += 1
    gainSum[j] += treedict['trees'][i][0]['value'][0]

total = 0
for sum in gainSum:
    if(total!=0):
        percent = (sum / total) * 100
        percentGain.append(percent)
    total += sum
    
fig, ax = plt.subplots()
ax.plot(treeNum, gains,marker="o")
i = 0
for value in cutValues:
    ax.plot([0,treeNum[-1]],[value,value],linestyle='dashed')
    if(i!=0):
        plt.text(0,value+10,str(gainSum[i])+" {:4.2f}".format(percentGain[i-1])+"%",fontsize=12)
    i += 1

ax.set(xlabel='Tree #', ylabel='Gain',
              title='Gain vs Tree')
ax.set_ylim([0, 3000])
ax.grid()

fig.savefig("Gain.png")
