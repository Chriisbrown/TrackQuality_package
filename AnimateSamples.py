from Models.GBDTTrackQualityModel import XGBoostClassifierModel
from Utils.Dataset import *
from Utils.util import *
from Utils.TreeVis import *
import numpy as np
from scipy.special import expit
import os
import matplotlib.animation as animation
plt.rcParams['image.cmap'] = 'seismic'

folder = "Degredation9"
modelfolder =  "NoDegredation"

plot_every_event = False

model = XGBoostClassifierModel()
model.load_data("Datasets/"+folder+"/"+folder+"_Zp/")
model.load_model("Projects/"+modelfolder+"/Models/",modelfolder)
#model.test()
# synth_model(model,sim=False,hdl=False,hls=False,cpp=True,onnx=False,
#                 test_events=10000,
#                 precisions=['ap_fixed<10,5>'],
#                 save_dir="Test/")

fps = 10

n_events = 1000
treedict = convert(model.model)
tree_output_array = np.zeros([n_events,6,10])
tree_input_array = np.zeros([n_events,1,7])
prediction_array = np.zeros([n_events,1,1])

for event in range(n_events):
    temp_output_array = np.zeros([60])
    
    accumulation = 0
    for i in range(treedict['n_trees']):
        value = evaluateTree(model.DataSet.X_test[event:event+1],treedict['trees'][i][0],model.training_features)
        temp_output_array[i] = value
        accumulation += value

    tree_output_array[event] = temp_output_array.reshape((6, 10))
    temp_input_array = [model.DataSet.X_test[event:event+1][i].values[0] for i in model.training_features]
    input_maxes = [7,1,7,6,2,15,15]
    temp_input_array = [temp_input_array[i]/input_maxes[i] for i in range(len(input_maxes)) ]
    tree_input_array[event] = np.expand_dims(temp_input_array, axis=0)
    prediction = np.expand_dims(expit(accumulation), axis=0)
    prediction_array[event] = np.expand_dims(prediction, axis=0)


    if prediction < 0.5:

        fig, axd = plt.subplot_mosaic(
        [["input", "prediction"],
        ["output", "output"]],
        layout="constrained",
        # "image" will contain a square image. We fine-tune the width so that
        # there is no excess horizontal or vertical margin around the image.
        width_ratios=[7, 1],
        height_ratios=[1, 6],
        )

        axd["input"].imshow(tree_input_array[event],vmin=-1,vmax=1)
        labels = "$|\\tan(\\lambda)|$","|$z_0$|","$\\chi^2_{bend}$","# Stubs","# Missing \n Internal Stubs","$\\chi^2_{r\\phi}$","$\\chi^2_{rz}$"
        label_pos = [-0.3,0.9,1.9,2.7,3.5,4.9,5.9]
        label_pos_y = [0.9,0.9,0.85,0.9,1.0,0.9,0.9]
        axd["input"].axis('off')
        for i,label in enumerate(labels):
            axd["input"].text(label_pos[i], label_pos_y[i], label, fontsize=15)    

        axd["prediction"].imshow(prediction_array[event],vmin=-1,vmax=1)
        axd["prediction"].text(-0.25, 0.9, "Output", fontsize=15)    
        axd["prediction"].axis('off')
        im = axd["output"].imshow(tree_output_array[event],vmin=-1,vmax=1)
        axd["output"].axis('off')
        axd["output"].text(3.5, 6, "Tree Output", fontsize=20)  

        plt.axis('off')
        plt.savefig("trees"+str(event)+".png",dpi=600)
        plt.clf()
        plt.close()

fig, axd = plt.subplot_mosaic(
        [["input", "prediction"],
        ["output", "output"]],
        layout="constrained",
        # "image" will contain a square image. We fine-tune the width so that
        # there is no excess horizontal or vertical margin around the image.
        width_ratios=[7, 1],
        height_ratios=[1, 6],
        )

reordered = np.argsort(prediction_array,axis=0)
tree_input_array  = tree_input_array[reordered[:,0,0]]
tree_output_array = tree_output_array[reordered[:,0,0]]
prediction_array  = prediction_array[reordered[:,0,0]]

im_i = axd["input"].imshow(tree_input_array[0],vmin=-1,vmax=1)
labels = "$|\\tan(\\lambda)|$","|$z_0$|","$\\chi^2_{bend}$","# Stubs","# Missing \n Internal Stubs","$\\chi^2_{r\\phi}$","$\\chi^2_{rz}$"
label_pos = [-0.3,0.9,1.9,2.7,3.5,4.9,5.9]
label_pos_y = [0.9,0.9,0.85,0.9,1.0,0.9,0.9]
axd["input"].axis('off')
for i,label in enumerate(labels):
    axd["input"].text(label_pos[i], label_pos_y[i], label, fontsize=15)    

im_p = axd["prediction"].imshow(prediction_array[0],vmin=-1,vmax=1)
axd["prediction"].text(-0.25, 0.9, "Output", fontsize=15)    
axd["prediction"].axis('off')
im_o = axd["output"].imshow(tree_output_array[0],vmin=-1,vmax=1)
axd["output"].axis('off')
axd["output"].text(3.5, 6, "Tree Output", fontsize=20)  


def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im_i.set_array(tree_input_array[i])
    im_p.set_array(prediction_array[i])
    im_o.set_array(tree_output_array[i])
    return [im_i,im_p,im_o]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = n_events,
                               interval = 1000 / fps, # in ms
                               )

anim.save('test_anim.mp4', fps=fps)

print('Done!')

# plt.show()  # Not required, it seems!


