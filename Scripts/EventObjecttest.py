from Datasets.Dataset import *

def predictFastHisto(value,weight):
    halfBinWidth = 0.5*30./256.
    hist,bin_edges = np.histogram(value,256,range=(-15,15),weights=weight)
    hist = np.convolve(hist,[1,1,1],mode='same')
    z0Index= np.argmax(hist)
    z0 = -15.+30.*z0Index/256.+halfBinWidth
    return np.array(z0,dtype=np.float32)

def predictMET(pt,phi,predictedAssoc,threshold):
    predictedAssoc[predictedAssoc > threshold] = 1
    NN_track_sel = predictedAssoc == 1


    met_px = np.sum(pt[NN_track_sel[:,0]]*np.cos(phi[NN_track_sel[:,0]]))
    met_py = np.sum(pt[NN_track_sel[:,0]]*np.sin(phi[NN_track_sel[:,0]]))

    return  [np.array(met_px,dtype=np.float32),
            np.array(met_py,dtype=np.float32),
            np.array(math.sqrt(met_px**2+met_py**2),dtype=np.float32),
            np.array(math.atan2(met_py,met_px),dtype=np.float32)]


def FastHistoAssoc(PV,trk_z0,trk_eta,threshold=1):
    eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])
    deltaz_bins = threshold*np.array([0.0,0.4,0.6,0.76,1.0,1.7,2.2,0.0])
    

    eta_bin = np.digitize(abs(trk_eta),eta_bins)
    deltaz = trk_z0 - PV

    assoc = (deltaz < deltaz_bins[eta_bin])

    return np.array(assoc,dtype=np.float32)


newKFKFDataset = KFEventDataSet("NewKFKF_Events")
newKFKFDataset.load_data_from_root("/home/cb719/Documents/Datasets/NewKF_object/NewKF_TTbar_300K",1000)

newKFTrackDataset = TrackEventDataSet("NewKFKF_Events")
newKFTrackDataset.load_data_from_root("/home/cb719/Documents/Datasets/NewKF_object/NewKF_TTbar_300K",1000)

oldKFTrackDataset = TrackEventDataSet("NewKFKF_Events")
oldKFTrackDataset.load_data_from_root("/home/cb719/Documents/Datasets/NewKF_object/OldKF_TTbar_300K",1000)