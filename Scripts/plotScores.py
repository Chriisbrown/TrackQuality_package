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
model = XGBoostClassifierModel("Degredation_0")
filepath = "/home/ryanm124/ml_hackathon/TrackQuality_package/Projects/Degradation0_Test/Models/Degradation0/"
model.load_model(filepath,"Degradation0_XGB")
model.load_data("/home/ryanm124/ml_hackathon/TrackQuality_package/Datasets/Degradation1_Train/")

treedict = convert(model.model.get_booster())
n_trees = treedict['n_trees']
x_tree = range(n_trees)
max_predict_fake = 0.0
min_predict_fake = 10.0
max_predict_real = 0.0
min_predict_real = 10.0
worst_fake_scores= np.zeros([n_trees])
worst_real_scores= np.zeros([n_trees])
best_fake_scores= np.zeros([n_trees])
best_real_scores= np.zeros([n_trees])
n_events = len(model.DataSet.X_train)
for event in range(10000):
    if(event%1000==0):
        print("event:",event)
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

fig, ax = plt.subplots()
ax.plot(x_tree, worst_real_scores,marker="o")
ax.set(xlabel='Tree #', ylabel='Output',
       title='Output vs Tree')
ax.grid()
fig.savefig("WorstReal.png")
plt.cla()
ax.plot(x_tree, worst_fake_scores,marker="o")
ax.set(xlabel='Tree #', ylabel='Output',
       title='Output vs Tree')
ax.grid()
fig.savefig("WorstFake.png")
plt.cla()
ax.plot(x_tree, best_real_scores,marker="o")
ax.set(xlabel='Tree #', ylabel='Output',
       title='Output vs Tree')
ax.grid()
fig.savefig("BestReal.png")
plt.cla()
ax.plot(x_tree, best_fake_scores,marker="o")
ax.set(xlabel='Tree #', ylabel='Output',
       title='Output vs Tree')
ax.grid()
fig.savefig("BestFake.png")
print("max_predict_fake",max_predict_fake)
print("min_predict_real",min_predict_real)
print("min_predict_fake",min_predict_fake)
print("max_predict_real",max_predict_real)
