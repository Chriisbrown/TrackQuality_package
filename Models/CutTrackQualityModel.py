from Datasets.Dataset import *
from sklearn import metrics
import numpy as np
import joblib
from pathlib import Path
import os


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mplhep as hep
#hep.set_style("CMSTex")
hep.cms.label()
hep.cms.text("Simulation")

plt.style.use(hep.cms.style.ROOT)

from Models.TrackQualityModel import TrackClassifierModel

class CutClassifierModel(TrackClassifierModel):
    def __init__(self,name):
        super().__init__(name)

        self.z0_cut = 15
        self.eta_cut = 2.4
        self.pt_cut = 2
        self.nstub_cut = 4
        self.chi2rphi_cut = 20
        self.chi2rz_cut = 20
        self.chi2_cut = 40
        self.bendchi2_cut = 2.2

        self.qchi2rphi_cut = 20
        self.qchi2rz_cut = 20
        self.qbendchi2_cut = 2.25

    def test(self):

        self.y_predict = np.where(((abs(self.DataSet.X_test["trk_z0"]) <= self.z0_cut) &
                                   (abs(self.DataSet.X_test["trk_eta"]) <= self.eta_cut)&
                                       (self.DataSet.X_test["trk_pt"] >= self.pt_cut) &
                                       (self.DataSet.X_test["trk_nstub"] >= self.nstub_cut) &
                                       (self.DataSet.X_test["trk_chi2"] <= self.chi2_cut) &
                                       (self.DataSet.X_test["trk_bendchi2"] <= self.bendchi2_cut))
                                      , 1, 0)

        self.y_predict_proba = self.y_predict


        self.q_y_predict = np.where(((abs(self.DataSet.X_test["trk_z0"]) <= self.z0_cut) &
                                     (abs(self.DataSet.X_test["trk_eta"]) <= self.eta_cut)&
                                         (self.DataSet.X_test["trk_pt"] >= self.pt_cut) &
                                         (self.DataSet.X_test["trk_nstub"] >= self.nstub_cut) &
                                         (np.digitize(self.DataSet.X_test["trk_chi2rphi"]/(self.DataSet.X_test["trk_nstub"] - 2),bins=trackword_config["Chi2rphi"]["bins"]) <= np.digitize(self.qchi2rphi_cut,bins=trackword_config["Chi2rphi"]["bins"])) &
                                         (np.digitize(self.DataSet.X_test["trk_chi2rz"]/(self.DataSet.X_test["trk_nstub"] - 2),bins=trackword_config["Chi2rz"]["bins"]) <= np.digitize(self.qchi2rz_cut,bins=trackword_config["Chi2rz"]["bins"])) &
                                         (np.digitize(self.DataSet.X_test["trk_bendchi2"],bins=trackword_config["Bendchi2"]["bins"]) <= np.digitize(self.qbendchi2_cut,bins=trackword_config["Bendchi2"]["bins"])))
                                      , 1, 0)

        self.y_predict_quantised = self.q_y_predict



class Binned_CutClassifierModel(TrackClassifierModel):
    def __init__(self,name):
        super().__init__(name)

        self.z0_cut = 15
        self.eta_cut = 2.4
        self.pt_cut = 2
        self.nstub_cut = 4
        self.chi2rphi_cut = 8
        self.chi2rz_cut = 14
        self.bendchi2_cut = 4

        #[TanL', 'RescaledAbsZ', 'bit_bendchi2', 'trk_nstub', 'nlay_miss', 'bit_chi2rphi', 'bit_chi2rz

    def test(self):

        self.y_predict = np.where(((abs(self.DataSet.X_test["trk_z0"]) <= self.z0_cut) &
                                   (abs(self.DataSet.X_test["trk_eta"]) <= self.eta_cut)&
                                       (self.DataSet.X_test["trk_pt"] >= self.pt_cut) &
                                       (self.DataSet.X_test["trk_nstub"] >= self.nstub_cut) &
                                       (self.DataSet.X_test["bit_chi2rphi"] <= self.chi2rphi_cut) &
                                       (self.DataSet.X_test["bit_chi2rz"] <= self.chi2rz_cut) &
                                       (self.DataSet.X_test["bit_bendchi2"] <= self.bendchi2_cut))
                                      , 1, 0)

        self.y_predict_proba = self.y_predict
        self.y_predict_binned = self.y_predict_proba



