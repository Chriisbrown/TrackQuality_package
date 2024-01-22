from Models.GBDTTrackQualityModel import XGBoostClassifierModel, TFDFClassifierModel
from Models.CutTrackQualityModel import CutClassifierModel
from Datasets.Dataset import *
from Utils.util import *
from Utils.TreeVis import *
import numpy as np
from scipy.special import expit
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def calcAvgDiff(scores):
    diffSum= 0.0
    for i in range(len(scores)-1):
        diffSum += abs(scores[i]-scores[i+1])
    return (diffSum/(len(scores)-1))


model = XGBoostClassifierModel("Degredation_0")
filepath = "/home/ryanm124/ml_hackathon/TrackQuality_package/Projects/Degradation0_Test/Models/Degradation0/"
model.load_model(filepath,"Degradation0_XGB")
folder_list = ["Degradation1","Degradation2","Degradation3","Degradation4","Degradation5","Degradation6","Degradation7","Degradation8","Degradation9","Degradation10"]
treedict = convert(model.model.get_booster())
n_trees = treedict['n_trees']
x_tree = range(n_trees)
avgDiffMean_worstReal = []
avgDiffMean_worstFake = []
avgDiffMean_bestReal = []
avgDiffMean_bestFake = []

for folder in folder_list:
    model.load_data("/home/ryanm124/ml_hackathon/TrackQuality_package/Datasets/"+folder+"_Train/")
    avgDiff_worstReal = []
    avgDiff_worstFake = []
    avgDiff_bestReal = []
    avgDiff_bestFake = []
    for i in range(10):
        max_predict_fake = 0.0
        min_predict_fake = 10.0
        max_predict_real = 0.0
        min_predict_real = 10.0
        worst_fake_scores= np.zeros([n_trees])
        worst_real_scores= np.zeros([n_trees])
        best_fake_scores= np.zeros([n_trees])
        best_real_scores= np.zeros([n_trees])
        n_events = len(model.DataSet.X_train)
        for event in range(i*1000,(i+1)*1000):
            temp_output_array = np.zeros([n_trees])
            accumulation = 0
            for i in range(treedict['n_trees']):
                value = evaluateTree(model.DataSet.X_train[event:event+1],treedict['trees'][i][0],model.training_features)
                temp_output_array[i] = value
                accumulation += value
    
            prediction = expit(accumulation)
            if(model.DataSet.y_train[event:event+1].values[0][0]==1 and prediction<min_predict_real):
                min_predict_real = prediction
                worst_real_scores = temp_output_array
            if(model.DataSet.y_train[event:event+1].values[0][0]==1 and prediction>max_predict_real):
                max_predict_real = prediction
                best_real_scores = temp_output_array
            if(model.DataSet.y_train[event:event+1].values[0][0]==0 and prediction>max_predict_fake):
                max_predict_fake = prediction
                worst_fake_scores = temp_output_array
            if(model.DataSet.y_train[event:event+1].values[0][0]==0 and prediction<min_predict_fake):
                min_predict_fake = prediction
                best_fake_scores = temp_output_array
        avgDiff_worstReal.append(calcAvgDiff(worst_real_scores[40:]))
        avgDiff_bestReal.append(calcAvgDiff(best_real_scores[40:]))
        avgDiff_worstFake.append(calcAvgDiff(worst_fake_scores[40:]))
        avgDiff_bestFake.append(calcAvgDiff(best_fake_scores[40:]))
    avgDiffMean_worstReal.append(sum(avgDiff_worstReal)/len(avgDiff_worstReal))
    avgDiffMean_worstFake.append(sum(avgDiff_worstFake)/len(avgDiff_worstFake))
    avgDiffMean_bestReal.append(sum(avgDiff_bestReal)/len(avgDiff_bestReal))
    avgDiffMean_bestFake.append(sum(avgDiff_bestFake)/len(avgDiff_bestFake))
fig, ax = plt.subplots()
x_samples = range(1,11)
ax.plot(x_samples, avgDiffMean_worstReal,marker="o",label="Real Track Lowest Score")
ax.plot(x_samples, avgDiffMean_bestReal,marker="o",label="Real Track Highest Score")
ax.plot(x_samples, avgDiffMean_worstFake,marker="o",label="Fake Track Highest Score")
ax.plot(x_samples, avgDiffMean_bestFake,marker="o",label="Fake Track Lowest Score")
ax.set(xlabel='Degredation %', ylabel='Avg Diff',
       title='Average Difference vs Degredation')
ax.grid()
ax.legend()
fig.savefig("AverageDiff.png")

