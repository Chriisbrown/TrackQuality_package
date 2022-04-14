import numpy as np
import uproot
import pandas as pd
import util
from sklearn.model_selection import train_test_split
import gc
import datetime
from pathlib import Path
import json
import warnings
import pickle
from Formats import *


trackword_config = {'InvR':      {'nbits': 15, 'granularity': 5.20424e-07, "Signed": True},
                    'Phi':       {'nbits': 12, 'granularity': 0.000340885, "Signed": True},
                    'TanL':      {'nbits': 16, 'granularity': 0.000244141, "Signed": True},
                    'Z0':        {'nbits': 12, 'granularity':  0.00999469, "Signed": True},
                    'D0':        {'nbits': 13, 'granularity': 3.757580e-3, "Signed": True},
                    'Chi2rphi':  {'nbins':2**4,'bins':[0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0, np.inf]},
                    'Chi2rz':    {'nbins':2**4,'bins':[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0,np.inf]},
                    'Bendchi2':  {'nbins':2**3,'bins':[0, 0.75, 1.0, 1.5, 2.25, 3.5, 5.0, 20.0,np.inf]},
                    'Hitpattern':{'nbins':2**7 ,'min':0,       'max':2**7,   "Signed":False,'split':(7 ,0)},
                    'MVA1':      {'nbins':2**3 ,'min':0,       'max':1,      "Signed":False,'split':(3 ,0)},
                    'OtherMVA':  {'nbins':2**6 ,'min':0,       'max':1,      "Signed":False,'split':(6 ,0)},
                    'TargetPrecision':{"full":13,"int":6}}


class DataSet:
    def __init__(self, name):

        self.name = name

        self.data_frame = pd.DataFrame()

        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()

        self.random_state = 4
        self.test_size = 0.1
        self.balance_type = "particle"
        self.verbose = 1

        self.config_dict = {"name": self.name,
                            "rootLoaded": None,
                            "Rootfilepath": "",
                            "datatransformed": False,
                            "databitted": False,
                            "datanormalised": False,
                            "balanced": "",
                            "testandtrain": False,
                            "testandtrainfilepath": "",
                            "h5filepath": "",
                            "NumEvents": 0,
                            "NumTracks": 0,
                            "NumTrainTracks": 0,
                            "NumTestTracks": 0,
                            "NumTrainFakes": 0,
                            "NumTrainElectrons": 0,
                            "NumTrainMuons": 0,
                            "NumTrainHadrons": 0,
                            "trainingFeatures": None,
                            "randomState": self.random_state,
                            "testsize": self.test_size,
                            "save_timestamp": None,
                            "loaded_timestamp": None
                            }

    @classmethod
    def fromROOT(cls, filepath, numEvents):
        rootclass = cls("From Root")
        rootclass.load_data_from_root(filepath=filepath, numEvents=numEvents)
        return rootclass

    @classmethod
    def fromH5(cls, filepath):
        h5class = cls("From H5")
        h5class.load_h5(filepath=filepath)
        return h5class

    @classmethod
    def fromTrainTest(cls, filepath):
        traintest = cls("From Train Test")
        traintest.load_test_train_h5(filepath=filepath)
        return traintest

    def load_data_from_root(self, filepath: str, numEvents: int, start : int = 0):
        tracks = uproot.open(
            filepath+".root:L1TrackNtuple;1/eventTree;1", numEvents=numEvents)
        TTTrackDF = pd.DataFrame()
        for array in tracks.iterate(library="pd",filter_name=self.feature_list,entry_start=start,entry_stop=start+numEvents):
            TTTrackDF = TTTrackDF.append(array,ignore_index=False)
            print("Cumulative tracks read: ", len(TTTrackDF))
        trackskept = self.feature_list

        Tracks = TTTrackDF[trackskept]
        Tracks.reset_index(inplace=True)
        Tracks.dropna(inplace=True)


        del [TTTrackDF]

        for j in trackskept:
            self.data_frame[j] = Tracks[j]
        del [Tracks]

        infs = np.where(np.asanyarray(np.isnan(self.data_frame)))[0]
        self.data_frame.drop(infs, inplace=True)
        self.config_dict["rootLoaded"] = datetime.datetime.now().strftime(
            "%H:%M %d/%m/%y")
        self.config_dict["Rootfilepath"] = filepath
        self.config_dict["NumEvents"] = numEvents
        self.config_dict["NumTracks"] = len(self.data_frame)
        if self.verbose == 1:
            print("Track Reading Complete, read: ",
                  len(self.data_frame), " tracks")

    def transform_data(self):
        pass

    def bit_data(self, normalise: bool = False):
        pass

    def fake_balance_data(self, fake_to_genuine_ratio: float = 1):

        genuine = util.genuineTracks(self.data_frame)
        numgenuine = len(genuine)

        fake = util.fakeTracks(self.data_frame)
        numfake = len(fake)

        self.config_dict["NumTrainTracks"] = numfake + numgenuine
        self.config_dict["NumTrainFakes"] = numfake
        self.config_dict["fakebalanced"] = True

        fraction = numfake/numgenuine*fake_to_genuine_ratio

        genuine = genuine.sample(
            frac=fraction, replace=True, random_state=self.random_state)

        self.data_frame = pd.concat([fake, genuine], ignore_index=True)

        del [fake, genuine]

        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)
        if self.verbose == 1:
            print("===Fake Balanced===")

    def particle_balance_data(self, e_multiplier: float = 0.495, h_multiplier: float = 0.495, f_multiplier: float = 1, m_mulitpler: float = 0.01):
        muons = self.data_frame[(self.data_frame["trk_matchtp_pdgid"] == -13)
                                | (self.data_frame["trk_matchtp_pdgid"] == 13)]
        electrons = self.data_frame[(
            self.data_frame["trk_matchtp_pdgid"] == -11) | (self.data_frame["trk_matchtp_pdgid"] == 11)]

        efraction = e_multiplier * len(muons)/(len(electrons) * m_mulitpler)
        electronsample = electrons.sample(
            frac=efraction, replace=True, random_state=self.random_state)
        del [electrons]
        print("Electrons")

        hadrons = self.data_frame[(self.data_frame["trk_matchtp_pdgid"] != -13) & (self.data_frame["trk_matchtp_pdgid"] != 13) & (
            self.data_frame["trk_matchtp_pdgid"] != 11) & (self.data_frame["trk_matchtp_pdgid"] != -11)]
        hfraction = h_multiplier * len(muons)/(len(hadrons) * m_mulitpler)
        hadronsample = hadrons.sample(
            frac=hfraction, replace=True, random_state=self.random_state)
        del [hadrons]
        print("Hadrons")

        fakes = self.data_frame[self.data_frame["trk_fake"] == 0]
        ffraction = f_multiplier * len(muons)/(len(fakes) * m_mulitpler)
        fakesample = fakes.sample(frac=ffraction, replace=True,
                             random_state=self.random_state)
        del [fakes]
        print("Fakes")
        del [ self.data_frame]

        self.config_dict["NumTrainFakes"] = len(fakesample)
        self.config_dict["NumTrainElectrons"] = len(electronsample)
        self.config_dict["NumTrainMuons"] = len(muons)
        self.config_dict["NumTrainHadrons"] = len(hadronsample)
        self.config_dict["NumTrainTracks"] = self.config_dict["NumTrainFakes"] + \
            self.config_dict["NumTrainElectrons"] + \
            self.config_dict["NumTrainMuons"] + \
            self.config_dict["NumTrainHadrons"]
        self.config_dict["particlebalanced"] = True

        self.data_frame = pd.concat(
            [electronsample, muons, hadronsample, fakesample], ignore_index=True)

        del [electronsample, muons, hadronsample, fakesample]
        gc.collect()

        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)
        if self.verbose == 1:
            print("===Particle Balanced===")

    def pt_balance_data(self, pt_bins=[2, 10, 100]):

        df_pt_bins = []
        pt_bin_widths = []

        pt_bins = [util.splitter(util.pttoR(
            pt), part_len=self.trackword_config["InvR"]['split']) for pt in pt_bins]
        for i in range(len(pt_bins)-1):
            temp_df = self.data_frame[(self.data_frame["bit_InvR"] <= pt_bins[i]) & (
                self.data_frame["bit_InvR"] > pt_bins[i+1])]
            pt_bin_widths.append(len(temp_df))

        fractions = [min(pt_bin_widths)/pt for pt in pt_bin_widths]

        for i in range(len(pt_bins)-1):
            temp_df = self.data_frame[(self.data_frame["bit_InvR"] <= pt_bins[i]) & (
                self.data_frame["bit_InvR"] > pt_bins[i+1])]
            temp_df = temp_df.sample(
                frac=fractions[i], replace=True, random_state=self.random_state)
            df_pt_bins.append(temp_df)

        self.data_frame = pd.concat(df_pt_bins, ignore_index=True)

        del [df_pt_bins, temp_df]
        gc.collect()
        if self.verbose == 1:
            print("===Pt Balanced===")
    
    def generate_test(self):
        self.transform_data()
        self.bit_data()

        self.X_test = self.data_frame[self.training_features]

        temp_y_test = np.array(self.data_frame["trk_fake"].values.tolist())
        self.y_test["label"] = np.where(
            temp_y_test > 0, 1, temp_y_test).tolist()

        self.config_dict["NumTestTracks"] = len(self.y_test)
        self.data_frame = None

        gc.collect()
        if self.verbose == 1:
            print("===Test Created====")

    def generate_test_train(self):

        self.transform_data()
        self.bit_data()

        training_features_extra = self.training_features.copy()
        training_features_extra.append("trk_fake")
        training_features_extra.append("trk_matchtp_pdgid")

        # Only want to particle balance the training data so we perform train test split, then merge train back together and then particle balance
        
        train, test = train_test_split( self.data_frame[training_features_extra], test_size=self.test_size, random_state=self.random_state)
        del [self.data_frame]

        self.data_frame = train

        if self.balance_type == "particle":
            self.particle_balance_data()
        elif self.balance_type == "fake":
            self.fake_balance_data()
        elif self.balance_type == "pt":
            self.pt_balance_data()
        else:
            warnings.warn("The Balance type "+self.balance_type +
                          " is unsupported. No balancing will be performed. Choose from particle, fake or pt. ", stacklevel=3)



        self.X_train = self.data_frame[self.training_features]

        temp_y_train = np.array(self.data_frame["trk_fake"].values.tolist())
        self.y_train["label"] = np.where(
            temp_y_train > 0, 1, temp_y_train).tolist()

        self.X_test = test[self.training_features]

        temp_y_test = np.array(test["trk_fake"].values.tolist())
        self.y_test["label"] = np.where(
            temp_y_test > 0, 1, temp_y_test).tolist()
        self.config_dict["NumTestTracks"] = len(self.y_test)
        self.data_frame = None
        del[train, test]
        gc.collect()
        self.config_dict["testandtrain"] = True
        if self.verbose == 1:
            print("===Train Test Split====")

    def save_h5(self, filepath):
        Path(filepath).mkdir(parents=True, exist_ok=True)

        store = pd.HDFStore(filepath+'full_Dataset.h5')
        store['df'] = self.data_frame  # save it
        self.data_frame = None
        store.close()

        self.config_dict["h5filepath"] = filepath
        self.config_dict["save_timestamp"] = datetime.datetime.now().strftime(
            "%H:%M %d/%m/%y")
        with open(filepath+'config_dict.json', 'w') as f:
            json.dump(self.config_dict, f, indent=4)
        if self.verbose == 1:
            print("===Full Data Saved===")

    def load_h5(self, filepath):
        my_file = Path(filepath+'full_Dataset.h5')
        if my_file.is_file():

            store = pd.HDFStore(filepath+'full_Dataset.h5')
            self.data_frame = store['df']
            store.close()
        else:
            print("No Full dataset")

        my_file2 = Path(filepath+'config_dict.json')
        if my_file2.is_file():
            with open(filepath+'config_dict.json', 'r') as f:
                self.config_dict = json.load(f)
            self.config_dict["loaded_timestamp"] = datetime.datetime.now().strftime(
                "%H:%M %d/%m/%y")
            self.name = self.config_dict["name"]
        else:
            print("No Config Dict")

    def save_test_train_h5(self, filepath):
        self.config_dict["balanced"] = self.balance_type
        self.config_dict["trainingFeatures"] = self.training_features
        Path(filepath).mkdir(parents=True, exist_ok=True)

        X_train_store = pd.HDFStore(filepath+'X_train.h5')
        X_test_store = pd.HDFStore(filepath+'X_test.h5')
        y_train_store = pd.HDFStore(filepath+'y_train.h5')
        y_test_store = pd.HDFStore(filepath+'y_test.h5')

        X_train_store['df'] = self.X_train  # save it
        X_test_store['df'] = self.X_test  # save it
        y_train_store['df'] = self.y_train  # save it
        y_test_store['df'] = self.y_test  # save it

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        X_train_store.close()
        X_test_store.close()
        y_train_store.close()
        y_test_store.close()

        self.config_dict["testandtrainfilepath"] = filepath
        self.config_dict["save_timestamp"] = datetime.datetime.now().strftime(
            "%H:%M %d/%m/%y")
        with open(filepath+'config_dict.json', 'w') as f:
            json.dump(self.config_dict, f, indent=4)

        if self.verbose == 1:
            print("===Train Test Saved====")

    def load_test_train_h5(self, filepath):
        X_train_file = Path(filepath+'X_train.h5')
        if X_train_file.is_file():
            X_train_store = pd.HDFStore(filepath+'X_train.h5')
            self.X_train = X_train_store['df']
            X_train_store.close()
        else:
            print("No X Train h5 File")

        X_test_file = Path(filepath+'X_test.h5')
        if X_test_file .is_file():
            X_test_store = pd.HDFStore(filepath+'X_test.h5')
            self.X_test = X_test_store['df']
            X_test_store.close()
        else:
            print("No X Test h5 File")

        y_train_file = Path(filepath+'y_train.h5')
        if y_train_file.is_file():
            y_train_store = pd.HDFStore(filepath+'y_train.h5')
            self.y_train = y_train_store['df']
            y_train_store.close()
        else:
            print("No y train h5 file")

        y_test_file = Path(filepath+'y_test.h5')
        if y_test_file.is_file():
            y_test_store = pd.HDFStore(filepath+'y_test.h5')
            self.y_test = y_test_store['df']
            y_test_store.close()
        else:
            print("No y test h5 file")

        config_dict_file = Path(filepath+'config_dict.json')
        if config_dict_file.is_file():
            with open(filepath+'config_dict.json', 'r') as f:
                self.config_dict = json.load(f)
            self.config_dict["loaded_timestamp"] = datetime.datetime.now().strftime(
                "%H:%M %d/%m/%y")
            self.name = self.config_dict["name"]
        else:
            print("No configuration dictionary json file")

    def __str__(self):
        print("=============================")
        print("Dataset Name: ", self.config_dict["name"])
        print("With random seed: ", self.config_dict["randomState"])
        print("Loaded From ROOT at: ", self.config_dict["rootLoaded"])
        print("from: ", self.config_dict["Rootfilepath"])
        print("Original Number of Events: ", self.config_dict["NumEvents"])
        print("Transformed          |", self.config_dict["datatransformed"])
        print("Bit formatted        |", self.config_dict["databitted"])
        print("Normalised           |", self.config_dict["datanormalised"])
        print("Balance type         |", self.config_dict["balanced"])
        print("Test and Train Split |", self.config_dict["testandtrain"])
        print("=============================")

        if (self.config_dict["balanced"] == "particle"):
            print("Test and Train Filepath: ",
                  self.config_dict["testandtrainfilepath"])
            print("Saved at: ", self.config_dict["save_timestamp"])
            print("Loaded at: ", self.config_dict["loaded_timestamp"])
            print("Total Tracks    |", self.config_dict["NumTracks"])
            print("Train Tracks    |", self.config_dict["NumTrainTracks"])
            print("Train Fakes     |", self.config_dict["NumTrainFakes"])
            print("Train Electrons |", self.config_dict["NumTrainElectrons"])
            print("Train Muons     |", self.config_dict["NumTrainMuons"])
            print("Train Hadrons   |", self.config_dict["NumTrainHadrons"])
            print("Total Test      |", self.config_dict["NumTestTracks"])
            print("Test Fraction   |", self.config_dict["testsize"])
            print("Training Features: ", self.config_dict["trainingFeatures"])

        else:
            print("h5 Filepath: ", self.config_dict["h5filepath"])
            print("Saved at: ", self.config_dict["save_timestamp"])
            print("Loaded at: ", self.config_dict["loaded_timestamp"])
        return "============================="

    def __add__(self, other):
        self.data_frame = self.data_frame.append(other.data_frame)
        self.config_dict["NumEvents"] = self.config_dict["NumEvents"] + other.config_dict["NumEvents"]
        self.config_dict["NumTracks"] = self.config_dict["NumTracks"] + other.config_dict["NumTracks"]
        return self

    def write_hls_file(self, filepath, num_events=10):
        self.__transform_data()
        self.__bit_data()
        Path(filepath).mkdir(parents=True, exist_ok=True)
        with open(filepath+"input_hls.txt", 'w') as f:
            for i in range(num_events):
                for j,feat in enumerate(self.training_features):
                    f.write("{j}".format(int(self.data_frame.iloc[i][feat])))
                f.write("\n")
        f.close()

class KFDataSet(DataSet):
    def __init__(self, name,withchi=False,withmatrix=False):
        super().__init__(name)

        # Track Word configuration as defined by https://twiki.cern.ch/twiki/bin/viewauth/CMS/HybridDataFormat#Fitted_Tracks_written_by_KalmanF
        self.trackword_config = {'InvR':      {'nbits': 16, 'granularity': 5.20424e-07, "Signed": True},
                                 'PhiT':      {'nbits': 11, 'granularity': 0.000340885, "Signed": True},
                                 'Cot':       {'nbits': 14, 'granularity': 0.000244141, "Signed": True},
                                 'ZT':        {'nbits': 13, 'granularity': 0.00999469,  "Signed": True},
                                 'r' :        {'nbits': 12, 'granularity': 0.0399788,   "Signed": False},
                                 'phi' :      {'nbits': 13, 'granularity': 4.26106e-05, "Signed": False},
                                 'z' :        {'nbits': 12, 'granularity': 0.0399788,   "Signed": False},
                                 'dPhi' :     {'nbits': 9,  'granularity': 4.26106e-05, "Signed": False},
                                 'dZ' :       {'nbits': 10, 'granularity': 0.0399788,   "Signed": False},
                                 'Chi2rphi':  {'nbins':2**4,'bins':[0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0, np.inf]},
                                 'Chi2rz':    {'nbins':2**4,'bins':[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0,np.inf]},
                                 'Bendchi2':  {'nbins':2**3,'bins':[0, 0.75, 1.0, 1.5, 2.25, 3.5, 5.0, 20.0,np.inf]},
                                 'C00' :       {'nbits': 16, 'granularity': 2.16674e-12,   "Signed": False},
                                 'C01' :       {'nbits': 17, 'granularity': 1.10878e-11,   "Signed": False},
                                 'C11' :       {'nbits': 16, 'granularity': 9.07831e-10,   "Signed": False},
                                 'C22' :       {'nbits': 16, 'granularity': 4.76837e-07,   "Signed": False},
                                 'C23' :       {'nbits': 17, 'granularity': 2.44011e-06,   "Signed": False},
                                 'C33' :       {'nbits': 16, 'granularity': 0.000199788,   "Signed": False},
                                 'TargetPrecision': {"full": 13, "int": 6}}

        self.withchi = withchi
        self.withmatrix = withmatrix
        # Set of branches extracted from track NTuple
        self.feature_list = ["KF_inv2R","KF_cot","KF_zT","KF_phiT",
                             "KF_stub_r_1","KF_stub_phi_1","KF_stub_z_1","KF_stub_dPhi_1","KF_stub_dZ_1","KF_stub_layer_1",
                             "KF_stub_r_2","KF_stub_phi_2","KF_stub_z_2","KF_stub_dPhi_2","KF_stub_dZ_2","KF_stub_layer_2",
                             "KF_stub_r_3","KF_stub_phi_3","KF_stub_z_3","KF_stub_dPhi_3","KF_stub_dZ_3","KF_stub_layer_3",
                             "KF_stub_r_4","KF_stub_phi_4","KF_stub_z_4","KF_stub_dPhi_4","KF_stub_dZ_4","KF_stub_layer_4",
                             "trk_fake", "trk_matchtp_pdgid"]
        if withchi:
            self.feature_list.extend(["trk_chi2rphi", "trk_chi2rz","trk_bendchi2"])
        if withmatrix:
            self.feature_list.extend(["KF_C00", "KF_C01","KF_C11","KF_C22", "KF_C23","KF_C33"])

        # Set of features used for training
        self.training_features = ["b_trk_inv2R","b_trk_cot","b_trk_zT","b_trk_phiT",
                                  "b_stub_r_1","b_stub_phi_1","b_stub_z_1","b_stub_dPhi_1","b_stub_dZ_1","b_stub_layer_1",
                                  "b_stub_r_2","b_stub_phi_2","b_stub_z_2","b_stub_dPhi_2","b_stub_dZ_2","b_stub_layer_2",
                                  "b_stub_r_3","b_stub_phi_3","b_stub_z_3","b_stub_dPhi_3","b_stub_dZ_3","b_stub_layer_3",
                                  "b_stub_r_4","b_stub_phi_4","b_stub_z_4","b_stub_dPhi_4","b_stub_dZ_4","b_stub_layer_4",
                                  ]

        if withchi:
            self.training_features.extend(["bit_bendchi2","bit_chi2rz","bit_chi2rphi"])
        if withmatrix:
            self.training_features.extend(["bit_C00", "bit_C01","bit_C11","bit_C22", "bit_C23","bit_C33"])
        # 0:trk_inv2R
        # 1:trk_cot
        # 2:trk_zT
        # 3:trk_phiT
        # 4:trk_match
        # 5:stub_r_0
        # 6:stub_phi_0
        # 7:stub_z_0
        # 8:stub_dPhi_0
        # 9:stub_dZ_0
        # 10:stub_layer_0
        # 11-17 : stub1
        # 18-23 : stub2
        # 24-29 : stub3
        # 30 : BendChi2
        # 31 : Chi2rz
        # 32 : Chi2rphi
        # (30) 33 : C00 
        # (31) 34 : C01 
        # (32) 35 : C11 
        # (33) 36 : C22 
        # (34) 37 : C23 
        # (35) 38 : C33 

    def bit_data(self, normalise: bool = False):

        self.data_frame.loc[:, "b_trk_inv2R"] = self.data_frame["KF_inv2R"].apply(util.splitter, granularity=self.trackword_config["InvR"]["granularity"],
                                                                                              signed=self.trackword_config["InvR"]["Signed"])                                                             
        self.data_frame.loc[:, "b_trk_cot"] = self.data_frame["KF_cot"].apply(util.splitter, granularity=self.trackword_config["Cot"]["granularity"],
                                                                                          signed=self.trackword_config["Cot"]["Signed"])
        self.data_frame.loc[:, "b_trk_zT"] = self.data_frame["KF_zT"].apply(util.splitter, granularity=self.trackword_config["ZT"]["granularity"],
                                                                                        signed=self.trackword_config["ZT"]["Signed"])
        self.data_frame.loc[:, "b_trk_phiT"] = self.data_frame["KF_phiT"].apply(util.splitter, granularity=self.trackword_config["PhiT"]["granularity"],
                                                                                            signed=self.trackword_config["PhiT"]["Signed"])

        for k in range(1,5):
            self.data_frame.loc[:, "b_stub_r_" + str(k)] =  self.data_frame["KF_stub_r_" + str(k)].apply(util.splitter, granularity=self.trackword_config["r"]["granularity"],
                                                                                              signed=self.trackword_config["r"]["Signed"])
            self.data_frame.loc[:, "b_stub_phi_" + str(k)]  = self.data_frame["KF_stub_phi_" + str(k)].apply(util.splitter, granularity=self.trackword_config["phi"]["granularity"],
                                                                                              signed=self.trackword_config["phi"]["Signed"])
            self.data_frame.loc[:, "b_stub_z_" + str(k)] =  self.data_frame["KF_stub_z_" + str(k)].apply(util.splitter, granularity=self.trackword_config["z"]["granularity"],
                                                                                              signed=self.trackword_config["z"]["Signed"])
            self.data_frame.loc[:, "b_stub_dPhi_" + str(k)]  = self.data_frame["KF_stub_dPhi_" + str(k)].apply(util.splitter, granularity=self.trackword_config["dPhi"]["granularity"],
                                                                                              signed=self.trackword_config["dPhi"]["Signed"])
            self.data_frame.loc[:, "b_stub_dZ_" + str(k)] =  self.data_frame["KF_stub_dZ_" + str(k)].apply(util.splitter, granularity=self.trackword_config["dZ"]["granularity"],
                                                                                              signed=self.trackword_config["dZ"]["Signed"])
            self.data_frame.loc[:, "b_stub_layer_" + str(k)]  = self.data_frame["KF_stub_layer_" + str(k)]

        if self.withchi:
            self.data_frame.loc[:,"bit_bendchi2"] = self.data_frame["trk_bendchi2"].apply(np.digitize,bins=self.trackword_config["Bendchi2"]["bins"])
            self.data_frame.loc[:,"bit_chi2rphi"] = self.data_frame["trk_chi2rphi"].apply(np.digitize,bins=self.trackword_config["Chi2rphi"]["bins"])
            self.data_frame.loc[:,"bit_chi2rz"]   = self.data_frame["trk_chi2rz"].apply(np.digitize,bins=self.trackword_config["Chi2rz"]["bins"])

        if self.withmatrix:
            names = ["C00", "C01","C11","C22", "C23","C33"]
            for name in names:
                self.data_frame.loc[:,"bit_"+name] = self.data_frame["KF_"+name].apply(util.splitter, granularity=self.trackword_config[name]["granularity"],
                                                                                                      signed=self.trackword_config[name]["Signed"])

        self.config_dict["databitted"] = True
        self.config_dict["datanormalised"] = normalise

        if normalise:
            mult = self.trackword_config["TargetPrecision"]["full"] - \
                self.trackword_config["TargetPrecision"]["int"]
            self.data_frame["b_trk_inv2R"] = self.data_frame["b_trk_inv2R"] * \
                (2**(mult-self.trackword_config["InvR"]["nbits"]))
            self.data_frame["b_trk_cot"] = self.data_frame["b_trk_cot"] * \
                (2**(mult-self.trackword_config["Cot"]["nbits"]))
            self.data_frame["b_trk_zT"] = self.data_frame["b_trk_zT"] * \
                (2**(mult-self.trackword_config["ZT"]["nbits"]))
            self.data_frame["b_trk_phiT"] = self.data_frame["b_trk_phiT"] * \
                (2**(mult-self.trackword_config["PhiT"]["nbits"]))

            for k in range(1,5):
                self.data_frame["b_stub_r_" + str(k)] = self.data_frame["b_stub_r_" + str(k)] * \
                (2**(mult-self.trackword_config["r"]["nbits"]))
                self.data_frame["b_stub_phi_" + str(k)] = self.data_frame["b_stub_phi_" + str(k)] * \
                (2**(mult-self.trackword_config["phi"]["nbits"]))
                self.data_frame["b_stub_z_" + str(k)] = self.data_frame["b_stub_z_" + str(k)] * \
                (2**(mult-self.trackword_config["z"]["nbits"]))
                self.data_frame["b_stub_dPhi_" + str(k)] = self.data_frame["b_stub_dPhi_" + str(k)] * \
                (2**(mult-self.trackword_config["dPhi"]["nbits"]))
                self.data_frame["b_stub_dZ_" + str(k)] = self.data_frame["b_stub_dZ_" + str(k)] * \
                (2**(mult-self.trackword_config["dZ"]["nbits"]))
            
            if self.withchi:
                self.data_frame["bit_bendchi2"] = self.data_frame["bit_bendchi2"]*(2**(mult-3))
                self.data_frame["bit_chi2rphi"] = self.data_frame["bit_chi2rphi"]*(2**(mult-4))
                self.data_frame["bit_chi2rz"] = self.data_frame["bit_chi2rz"]*(2**(mult-4))

            if self.withmatrix:
                names = ["C00", "C01","C11","C22", "C23","C33"]
                for name in names:
                    self.data_frame["bit_"+name] = self.data_frame["bit_"+name] * (2**(mult-self.trackword_config[name]["nbits"]))

            if self.verbose == 1:
                print("=======Normalised======")
        if self.verbose == 1:
            print("=====Bit Formatted=====")

class TrackDataSet(DataSet):
    def __init__(self, name):
        super().__init__(name)


        # Track Word configuration as defined by https://twiki.cern.ch/twiki/bin/viewauth/CMS/HybridDataFormat#Fitted_Tracks_written_by_KalmanF
        self.trackword_config = trackword_config
        # Set of branches extracted from track NTuple
        self.feature_list = ["trk_pt","trk_eta","trk_phi",
                             "trk_d0","trk_z0","trk_chi2rphi",
                             "trk_chi2rz","trk_bendchi2","trk_hitpattern",
                             "trk_fake","trk_matchtp_pdgid"]

        # Set of features used for training
        self.training_features = ["bit_TanL","bit_z0","bit_bendchi2","nlay_miss","bit_chi2rz","bit_chi2rphi","bit_chi2","bit_phi"]

    def transform_data(self):
        self.data_frame["InvR"] = self.data_frame["trk_pt"].apply(util.pttoR)
        self.data_frame["TanL"] = self.data_frame["trk_eta"].apply(util.tanL)
        self.data_frame = util.predhitpattern(self.data_frame)
        self.data_frame["nlay_miss"] = self.data_frame["trk_hitpattern"].apply(util.set_nlaymissinterior)
        self.config_dict["datatransformed"] = True
        if self.verbose == 1 : print("======Transfromed======")

    def bit_data(self,normalise : bool = False):
  
      self.data_frame.loc[:,"bit_bendchi2"] = self.data_frame["trk_bendchi2"].apply(np.digitize,bins=self.trackword_config["Bendchi2"]["bins"])
      self.data_frame.loc[:,"bit_chi2rphi"] = self.data_frame["trk_chi2rphi"].apply(np.digitize,bins=self.trackword_config["Chi2rphi"]["bins"])
      self.data_frame.loc[:,"bit_chi2rz"]   = self.data_frame["trk_chi2rz"].apply(np.digitize,bins=self.trackword_config["Chi2rz"]["bins"])
      self.data_frame.loc[:,"bit_phi"]      = self.data_frame["trk_phi"].apply(util.splitter, granularity=self.trackword_config["Phi"]["granularity"],
                                                                                              signed=self.trackword_config["Phi"]["Signed"])       
      self.data_frame["bit_chi2"]           = self.data_frame["bit_chi2rphi"]+self.data_frame["bit_chi2rz"]
      self.data_frame.loc[:,"bit_TanL"]     = self.data_frame["TanL"].apply(util.splitter, granularity=self.trackword_config["TanL"]["granularity"],
                                                                                              signed=self.trackword_config["TanL"]["Signed"])       
      self.data_frame.loc[:,"bit_z0"]       = self.data_frame["trk_z0"].apply(util.splitter, granularity=self.trackword_config["Z0"]["granularity"],
                                                                                              signed=self.trackword_config["Z0"]["Signed"])       
      self.data_frame.loc[:,"bit_d0"]       = self.data_frame["trk_d0"].apply(util.splitter, granularity=self.trackword_config["D0"]["granularity"],
                                                                                              signed=self.trackword_config["D0"]["Signed"])       

      self.data_frame.loc[:,"bit_InvR"]     = self.data_frame["InvR"].apply(util.splitter, granularity=self.trackword_config["InvR"]["granularity"],
                                                                                              signed=self.trackword_config["InvR"]["Signed"])       

      self.config_dict["databitted"] = True
      self.config_dict["datanormalised"] = normalise 

      if normalise:
        mult = self.trackword_config["TargetPrecision"]["full"]  - self.trackword_config["TargetPrecision"]["int"] 
        self.data_frame["bit_bendchi2"] = self.data_frame["bit_bendchi2"]*(2**(mult-3))
        self.data_frame["bit_chi2rphi"] = self.data_frame["bit_chi2rphi"]*(2**(mult-4))
        self.data_frame["bit_chi2rz"] = self.data_frame["bit_chi2rz"]*(2**(mult-4))
        self.data_frame["bit_TanL"] = self.data_frame["bit_TanL"]*(2**(mult-16))
        self.data_frame["bit_z0"] = self.data_frame["bit_z0"]*(2**(mult-12))
        self.data_frame["bit_InvR"] = self.data_frame["bit_InvR"]*(2**(mult-15))
        self.data_frame["pred_nstub"] = self.data_frame["pred_nstub"]*(2**(mult-3))
        self.data_frame["nlay_miss"] = self.data_frame["nlay_miss"]*(2**mult-2**3)

        if self.verbose == 1 : print("=======Normalised======")
      if self.verbose == 1 : print  ("=====Bit Formatted=====")

class FloatingTrackDataSet(DataSet):
    def __init__(self, name):
        super().__init__(name)



        # Set of branches extracted from track NTuple
        self.feature_list = ["trk_pt","trk_eta","trk_phi",
                             "trk_d0","trk_z0","trk_chi2rphi",
                             "trk_chi2rz","trk_bendchi2","trk_hitpattern",
                             "trk_fake","trk_matchtp_pdgid","trk_chi2","trk_nstub"]


        # Set of features used for training
        self.training_features = ["trk_pt","trk_eta","trk_z0","trk_bendchi2","trk_chi2rphi","trk_chi2rz","trk_chi2","trk_nstub"]

    def transform_data(self):
      self.config_dict["datatransformed"] = True
      if self.verbose == 1 : print("======Transfromed======")

    def bit_data(self,normalise : bool = False):
      if self.verbose == 1 : print("=======Normalised======")
      if self.verbose == 1 : print  ("=====Bit Formatted=====") 

class KFEventDataSet(KFDataSet):
    def __init__(self, name):
        super().__init__(name)

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.config_dict = {"name": self.name,
                            "rootLoaded": None,
                            "Rootfilepath": "",
                            "datatransformed": False,
                            "PV_found": False,
                            "testandtrain": False,
                            "testandtrainfilepath": "",
                            "h5filepath": "",
                            "NumEvents": 0,
                            "trainingFeatures": self.training_features,
                            "randomState": self.random_state,
                            "testsize": self.test_size,
                            "save_timestamp": None,
                            "loaded_timestamp": None
                            }

        self.feature_list.append("pv_MC","genMETPx","genMETPy","genMET","genMETPhi")

    def __bit_data(self, normalise=True):
        for event in self.data_frame:

            event.loc[:, "b_trk_inv2R"] = event["KFtrk_inv2R"].apply(util.splitter, granularity=self.trackword_config["InvR"]["granularity"],
                                                                                              signed=self.trackword_config["InvR"]["Signed"])                                                             
            event.loc[:, "b_trk_cot"] = event["KFtrk_cot"].apply(util.splitter, granularity=self.trackword_config["Cot"]["granularity"],
                                                                                            signed=self.trackword_config["Cot"]["Signed"])
            event.loc[:, "b_trk_zT"] = event["KFtrk_zT"].apply(util.splitter, granularity=self.trackword_config["ZT"]["granularity"],
                                                                                            signed=self.trackword_config["ZT"]["Signed"])
            event.loc[:, "b_trk_phiT"] = event["KFtrk_phiT"].apply(util.splitter, granularity=self.trackword_config["PhiT"]["granularity"],
                                                                                                signed=self.trackword_config["PhiT"]["Signed"])

            for k in range(1,5):
                event.loc[:, "b_stub_r_" + str(k)] =  event["KFtrk_r" + str(k)].apply(util.splitter, granularity=self.trackword_config["r"]["granularity"],
                                                                                                signed=self.trackword_config["r"]["Signed"])
                event.loc[:, "b_stub_phi_" + str(k)]  = event["KFtrk_phi" + str(k)].apply(util.splitter, granularity=self.trackword_config["phi"]["granularity"],
                                                                                                signed=self.trackword_config["phi"]["Signed"])
                event.loc[:, "b_stub_z_" + str(k)] =  event["KFtrk_z" + str(k)].apply(util.splitter, granularity=self.trackword_config["z"]["granularity"],
                                                                                                signed=self.trackword_config["z"]["Signed"])
                event.loc[:, "b_stub_dPhi_" + str(k)]  = event["KFtrk_dPhi" + str(k)].apply(util.splitter, granularity=self.trackword_config["dPhi"]["granularity"],
                                                                                                signed=self.trackword_config["dPhi"]["Signed"])
                event.loc[:, "b_stub_dZ_" + str(k)] =  event["KFtrk_dZ_" + str(k)].apply(util.splitter, granularity=self.trackword_config["dZ"]["granularity"],
                                                                                                signed=self.trackword_config["dZ"]["Signed"])
                event.loc[:, "b_stub_layer_" + str(k)]  = event["KFtrk_layer_" + str(k)]

            self.config_dict["databitted"] = True
            self.config_dict["datanormalised"] = normalise

            if normalise:
                mult = self.trackword_config["TargetPrecision"]["full"] - \
                    self.trackword_config["TargetPrecision"]["int"]
                event["b_trk_inv2R"] = event["b_trk_inv2R"] * \
                    (2**(mult-self.trackword_config["InvR"]["nbits"]))
                event["b_trk_cot"] = event["b_trk_cot"] * \
                    (2**(mult-self.trackword_config["Cot"]["nbits"]))
                event["b_trk_zT"] = event["b_trk_zT"] * \
                    (2**(mult-self.trackword_config["ZT"]["nbits"]))
                event["b_trk_phiT"] = event["b_trk_phiT"] * \
                    (2**(mult-self.trackword_config["PhiT"]["nbits"]))

                for k in range(1,5):
                    event["b_stub_r_" + str(k)] = event["b_stub_r_" + str(k)] * \
                    (2**(mult-self.trackword_config["r"]["nbits"]))
                    event["b_stub_phi_" + str(k)] = event["b_stub_phi_" + str(k)] * \
                    (2**(mult-self.trackword_config["phi"]["nbits"]))
                    event["b_stub_z_" + str(k)] = event["b_stub_z_" + str(k)] * \
                    (2**(mult-self.trackword_config["z"]["nbits"]))
                    event["b_stub_dPhi_" + str(k)] = event["b_stub_dPhi_" + str(k)] * \
                    (2**(mult-self.trackword_config["dPhi"]["nbits"]))
                    event["b_stub_dZ_" + str(k)] = event["b_stub_dZ_" + str(k)] * \
                    (2**(mult-self.trackword_config["dZ"]["nbits"]))

                if self.verbose == 1:
                    print("=======Normalised======")
            if self.verbose == 1:
                print("=====Bit Formatted=====")

    def load_data_from_root(self, filepath, numEvents):

        data = uproot.open(filepath+".root:L1TrackNtuple;1/eventTree;1",
                           numEvents=numEvents).arrays(self.feature_list)

        events = {}
        for branch in self.feature_list:
            events[branch] = []

        for branch in self.feature_list:
            for event in data[branch]:
                events[branch].append(event)

        # Pivot from dict of lists of arrays to list of dicts of arrays
        x = []
        for i in range(numEvents):
            y = {}
            for branch in self.feature_list:
                y[branch] = events[branch][i]
            x.append(pd.DataFrame(self.__transform_data(y)))
        events = x

        del [y, data]
        gc.collect()
        self.data_frame = events
        self.__bit_data()
        self.config_dict["NumEvents"] = len(events)

    def generate_test_train(self):

        # Only want to particle balance the training data so we perform train test split, then merge train back together and then particle balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_frame, self.pv_list, test_size=self.test_size, random_state=self.random_state)

        self.config_dict["NumTestEvents"] = len(self.y_test)
        self.config_dict["NumTrainEvents"] = len(self.y_train)
        self.data_frame = None
        self.pv_list = None

        self.config_dict["testandtrain"] = True
        if self.verbose == 1:
            print("===Train Test Split====")

    def __str__(self):
        print("=============================")
        print("Dataset Name: ", self.config_dict["name"])
        print("With random seed: ", self.config_dict["randomState"])
        print("Loaded From ROOT at: ", self.config_dict["rootLoaded"])
        print("from: ", self.config_dict["Rootfilepath"])
        print("Original Number of Events: ", self.config_dict["NumEvents"])
        print("Transformed          |", self.config_dict["datatransformed"])
        print("PVs Found            |", self.config_dict["PV_found"])
        print("Test and Train Split |", self.config_dict["testandtrain"])
        print("=============================")
        if self.config_dict["testandtrain"]:
            print("Test and Train Filepath: ",
                  self.config_dict["testandtrainfilepath"])
            print("Saved at: ", self.config_dict["save_timestamp"])
            print("Loaded at: ", self.config_dict["loaded_timestamp"])
            print("Test Tracks     |", self.config_dict["NumTrainEvents"])
            print("Test Events     |", self.config_dict["NumTestEvents"])
            print("Test Fraction   |", self.config_dict["testsize"])
            print("Training Features: ", self.config_dict["trainingFeatures"])
        return "============================="
      
class TrackEventDataSet(TrackDataSet):
    def __init__(self, name):
        super().__init__(name)

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.config_dict = {"name": self.name,
                            "rootLoaded": None,
                            "Rootfilepath": "",
                            "datatransformed": False,
                            "PV_found": False,
                            "testandtrain": False,
                            "testandtrainfilepath": "",
                            "h5filepath": "",
                            "NumEvents": 0,
                            "trainingFeatures": self.training_features,
                            "randomState": self.random_state,
                            "testsize": self.test_size,
                            "save_timestamp": None,
                            "loaded_timestamp": None
                            }

        self.feature_list.append("pv_MC","genMETPx","genMETPy","genMET","genMETPhi")
    def transform_data(self):
        for event in self.data_frame:
            event["InvR"] = event["trk_pt"].apply(util.pttoR)
            event["TanL"] = event["trk_eta"].apply(util.tanL)
            event = util.predhitpattern(event)
            event["nlay_miss"] = event["trk_hitpattern"].apply(util.set_nlaymissinterior)
        self.config_dict["datatransformed"] = True
        if self.verbose == 1 : print("======Transfromed======")

    def __bit_data(self, normalise=True):
        self.transform_data()
        for event in self.data_frame:

            event.loc[:,"bit_bendchi2"] = event["trk_bendchi2"].apply(np.digitize,bins=self.trackword_config["Bendchi2"]["bins"])
            event.loc[:,"bit_chi2rphi"] = event["trk_chi2rphi"].apply(np.digitize,bins=self.trackword_config["Chi2rphi"]["bins"])
            event.loc[:,"bit_chi2rz"]   = event["trk_chi2rz"].apply(np.digitize,bins=self.trackword_config["Chi2rz"]["bins"])
            event.loc[:,"bit_phi"]      = event["trk_phi"].apply(util.splitter, granularity=self.trackword_config["Phi"]["granularity"],
                                                                                                    signed=self.trackword_config["Phi"]["Signed"])       
            event["bit_chi2"]           = event["bit_chi2rphi"]+event["bit_chi2rz"]
            event.loc[:,"bit_TanL"]     = event["TanL"].apply(util.splitter, granularity=self.trackword_config["TanL"]["granularity"],
                                                                                                    signed=self.trackword_config["TanL"]["Signed"])       
            event.loc[:,"bit_z0"]       = event["trk_z0"].apply(util.splitter, granularity=self.trackword_config["Z0"]["granularity"],
                                                                                                    signed=self.trackword_config["Z0"]["Signed"])       
            event.loc[:,"bit_d0"]       = event["trk_d0"].apply(util.splitter, granularity=self.trackword_config["D0"]["granularity"],
                                                                                                    signed=self.trackword_config["D0"]["Signed"])       

            event.loc[:,"bit_InvR"]     = event["InvR"].apply(util.splitter, granularity=self.trackword_config["InvR"]["granularity"],
                                                                                                    signed=self.trackword_config["InvR"]["Signed"])       

            self.config_dict["databitted"] = True
            self.config_dict["datanormalised"] = normalise 

            if normalise:
                mult = self.trackword_config["TargetPrecision"]["full"]  - self.trackword_config["TargetPrecision"]["int"] 
                event["bit_bendchi2"] = event["bit_bendchi2"]*(2**(mult-3))
                event["bit_chi2rphi"] = event["bit_chi2rphi"]*(2**(mult-4))
                event["bit_chi2rz"] = event["bit_chi2rz"]*(2**(mult-4))
                event["bit_TanL"] = event["bit_TanL"]*(2**(mult-16))
                event["bit_z0"] = event["bit_z0"]*(2**(mult-12))
                event["bit_InvR"] = event["bit_InvR"]*(2**(mult-15))
                event["pred_nstub"] = event["pred_nstub"]*(2**(mult-3))
                event["nlay_miss"] = event["nlay_miss"]*(2**mult-2**3)

                if self.verbose == 1:
                    print("=======Normalised======")
            if self.verbose == 1:
                print("=====Bit Formatted=====")

    def load_data_from_root(self, filepath, numEvents):

        data = uproot.open(filepath+".root:L1TrackNtuple;1/eventTree;1",
                           numEvents=numEvents).arrays(self.feature_list)

        events = {}
        for branch in self.feature_list:
            events[branch] = []

        for branch in self.feature_list:
            for event in data[branch]:
                events[branch].append(event)

        # Pivot from dict of lists of arrays to list of dicts of arrays
        x = []
        for i in range(numEvents):
            y = {}
            for branch in self.feature_list:
                y[branch] = events[branch][i]
            x.append(pd.DataFrame(self.__bit_data(y,normalise=False)))
        events = x

        del [y, data]
        gc.collect()
        self.data_frame = events
        self.__bit_data()
        self.config_dict["NumEvents"] = len(events)

    def generate_test_train(self):

        # Only want to particle balance the training data so we perform train test split, then merge train back together and then particle balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_frame, self.pv_list, test_size=self.test_size, random_state=self.random_state)

        self.config_dict["NumTestEvents"] = len(self.y_test)
        self.config_dict["NumTrainEvents"] = len(self.y_train)
        self.data_frame = None
        self.pv_list = None

        self.config_dict["testandtrain"] = True
        if self.verbose == 1:
            print("===Train Test Split====")

    def __str__(self):
        print("=============================")
        print("Dataset Name: ", self.config_dict["name"])
        print("With random seed: ", self.config_dict["randomState"])
        print("Loaded From ROOT at: ", self.config_dict["rootLoaded"])
        print("from: ", self.config_dict["Rootfilepath"])
        print("Original Number of Events: ", self.config_dict["NumEvents"])
        print("Transformed          |", self.config_dict["datatransformed"])
        print("PVs Found            |", self.config_dict["PV_found"])
        print("Test and Train Split |", self.config_dict["testandtrain"])
        print("=============================")
        if self.config_dict["testandtrain"]:
            print("Test and Train Filepath: ",
                  self.config_dict["testandtrainfilepath"])
            print("Saved at: ", self.config_dict["save_timestamp"])
            print("Loaded at: ", self.config_dict["loaded_timestamp"])
            print("Test Tracks     |", self.config_dict["NumTrainEvents"])
            print("Test Events     |", self.config_dict["NumTestEvents"])
            print("Test Fraction   |", self.config_dict["testsize"])
            print("Training Features: ", self.config_dict["trainingFeatures"])
        return "============================="
