import numpy as np

def pttoR(pt):
    B = 3.8112 #Tesla for CMS magnetic field
    return abs((B*(3e8/1e11))/(2*pt))

def tanL(eta):
    return abs(np.sinh(eta))

def predhitpattern(dataframe):
    
        
    hitpat = dataframe["trk_hitpattern"].apply(bin).astype(str).to_numpy()
    eta = dataframe["trk_eta"].to_numpy()
    
    hit_array = np.zeros([7,len(hitpat)])
    expanded_hit_array = np.zeros([12,len(hitpat)])
    ltot = np.zeros(len(hitpat))
    dtot = np.zeros(len(hitpat))
    for i in range(len(hitpat)):
        for k in range(len(hitpat[i])-2):
            hit_array[k,i] = hitpat[i][-(k+1)]

    eta_bins = [0.0,0.2,0.41,0.62,0.9,1.26,1.68,2.08,2.5]
    conversion_table = np.array([[0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  3,  4,  5,  11],
                                 [0, 1,  2,  6,  7,  8,  9 ],
                                 [0, 1,  7,  8,  9, 10,  11],
                                 [0, 6,  7,  8,  9, 10,  11]])
   
    for i in range(len(hitpat)):
        for j in range(8):
            if ((abs(eta[i]) >= eta_bins[j]) & (abs(eta[i]) < eta_bins[j+1])):
                for k in range(7):
                    expanded_hit_array[conversion_table[j][k]][i] = hit_array[k][i]


        ltot[i] = sum(expanded_hit_array[0:6,i])
        dtot[i] = sum(expanded_hit_array[6:11,i])
    
    dataframe["pred_layer1"] = expanded_hit_array[0,:]
    dataframe["pred_layer2"] = expanded_hit_array[1,:]
    dataframe["pred_layer3"] = expanded_hit_array[2,:]
    dataframe["pred_layer4"] = expanded_hit_array[3,:]
    dataframe["pred_layer5"] = expanded_hit_array[4,:]
    dataframe["pred_layer6"] = expanded_hit_array[5,:]
    dataframe["pred_disk1"] = expanded_hit_array[6,:]
    dataframe["pred_disk2"] = expanded_hit_array[7,:]
    dataframe["pred_disk3"] = expanded_hit_array[8,:]
    dataframe["pred_disk4"] = expanded_hit_array[9,:]
    dataframe["pred_disk5"] = expanded_hit_array[10,:]
    dataframe["pred_ltot"] = ltot
    dataframe["pred_dtot"] = dtot
    dataframe["pred_nstub"] = ltot + dtot

    del [hit_array,hitpat,expanded_hit_array,ltot,dtot]

    return dataframe

def PVTracks(dataframe):
    return (dataframe[dataframe["trk_fake"]==1])

def pileupTracks(dataframe):
    return (dataframe[dataframe["trk_fake"]==2])

def fakeTracks(dataframe):
    return (dataframe[dataframe["trk_fake"]==0])

def genuineTracks(dataframe):
    return (dataframe[dataframe["trk_fake"] != 0])

def set_nlaymissinterior(hitpat):
    bin_hitpat = np.binary_repr(hitpat)
    bin_hitpat = bin_hitpat.strip('0')
    return bin_hitpat.count('0')

def nstub(hitpattern):
    bin_hitpat = hitpattern[2:]
    bin_hitpat = bin_hitpat.strip('0')
    return bin_hitpat.count('1')
   
def splitter(x,granularity,signed):
    import math

    mult = 1

    if signed: 
        mult = 2
      # Get the bin index
    t = (mult*x)/granularity

    return math.floor(t)

