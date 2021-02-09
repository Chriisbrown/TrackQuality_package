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

class DataSet:
    def __init__(self,name):

        self.name = name
        # Track Word configuration as defined by https://twiki.cern.ch/twiki/bin/viewauth/CMS/HybridDataFormat#Fitted_Tracks_written_by_KalmanF
        self.trackword_config = {'InvR':      {'nbins':2**15,'min':-0.00855,'max':0.00855,"Signed":True ,'split':(15,0)},
                                 'Phi':       {'nbins':2**12,'min':-1.026,  'max':1.026,  "Signed":False,'split':(11,0)},
                                 'TanL':      {'nbins':2**16,'min':-7,      'max':7,      "Signed":True, 'split':(16,3)},
                                 'Z0':        {'nbins':2**12,'min':-31,     'max':31,     "Signed":True, 'split':(12,5)},
                                 'D0':        {'nbins':2**13,'min':-15.4,   'max':15.4,   "Signed":True, 'split':(13,5)},
                                 'Chi2rphi':  {'bins':[0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 40, 100, 200, 500, 1000, 3000,np.inf]},
                                 'Chi2rz':    {'bins':[0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 40, 100, 200, 500, 1000, 3000,np.inf]},
                                 'Bendchi2':  {'bins':[0,0.5,1.25,2,3,5,10,50,np.inf]},
                                 'Hitpattern':{'nbins':2**7 ,'min':0,       'max':2**7,   "Signed":False,'split':(7 ,0)},
                                 'MVA1':      {'nbins':2**3 ,'min':0,       'max':1,      "Signed":False,'split':(3 ,0)},
                                 'OtherMVA':  {'nbins':2**6 ,'min':0,       'max':1,      "Signed":False,'split':(6 ,0)}}
        # Set of branches extracted from track NTuple
        self.feature_list = ["trk_pt","trk_eta","trk_phi",
                             "trk_d0","trk_z0","trk_chi2rphi",
                             "trk_chi2rz","trk_bendchi2","trk_hitpattern",
                             "trk_fake","trk_matchtp_pdgid"]

        # Set of features used for training
        self.training_features = ["bit_chi2","bit_bendchi2","bit_chi2rphi","bit_chi2rz","pred_nstub",
                                  "pred_layer1","pred_layer2","pred_layer3","pred_layer4","pred_layer5",
                                  "pred_layer6","pred_disk1","pred_disk2","pred_disk3","pred_disk4","pred_disk5",
                                  "bit_InvR","bit_TanL","bit_z0","pred_dtot","pred_ltot"]

                                  #0:bit_chi2
                                  #1:bit_bendchi2
                                  #2:bit_chi2rphi
                                  #3:bit_chi2rz
                                  #4:pred_nstub
                                  #5 - 15: pred_hitpattern
                                  #16:bit_InvR
                                  #17:bit_TanL
                                  #18:bit_z0
                                  #19:pred_dtot
                                  #20:pred_ltot
        self.data_frame = pd.DataFrame()

        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()

        self.random_state = 4
        self.test_size = 0.1
        self.balance_type = "particle"
        self.verbose = 0

        self.config_dict = {"name":self.name,
                            "rootLoaded":None,
                            "Rootfilepath":"",
                            "datatransformed":False,
                            "databitted":False,
                            "datanormalised":False,
                            "balanced":"",
                            "testandtrain":False,
                            "testandtrainfilepath":"",
                            "h5filepath":"",
                            "NumEvents":0,
                            "NumTracks":0,
                            "NumTrainTracks":0,
                            "NumTestTracks":0,
                            "NumTrainFakes":0,
                            "NumTrainElectrons":0,
                            "NumTrainMuons":0,
                            "NumTrainHadrons":0,
                            "trainingFeatures":None,
                            "randomState":self.random_state,
                            "testsize":self.test_size,
                            "save_timestamp":None,
                            "loaded_timestamp":None
                           }

    @classmethod
    def fromROOT(cls,filepath,numEvents):
        rootclass = cls("From Root")
        rootclass.load_data_from_root(filepath=filepath,numEvents=numEvents)
        return rootclass

    @classmethod
    def fromH5(cls,filepath):
        h5class = cls("From H5")
        h5class.load_h5(filepath=filepath)
        return h5class
    
    @classmethod
    def fromTrainTest(cls,filepath):
        traintest = cls("From Train Test")
        traintest.load_test_train_h5(filepath=filepath)
        return traintest

    def load_data_from_root(self,filepath : str,numEvents : int):
        tracks = uproot.open(filepath+".root:L1TrackNtuple;1/eventTree;1",numEvents=numEvents)
        for array in tracks.iterate(filter_name=self.feature_list,library="pd",entry_start=0,entry_stop=numEvents):
            self.data_frame = self.data_frame.append(array,ignore_index=True)
            if self.verbose == 1 : print("Cumulative tracks read: ",len(self.data_frame))

        infs = np.where(np.asanyarray(np.isnan(self.data_frame)))[0]
        self.data_frame.drop(infs,inplace=True)
        self.config_dict["rootLoaded"] = datetime.datetime.now().strftime("%H:%M %d/%m/%y")
        self.config_dict["Rootfilepath"] = filepath
        self.config_dict["NumEvents"] = numEvents
        self.config_dict["NumTracks"] = len(self.data_frame)
        if self.verbose == 1 : print("Track Reading Complete, read: ",len(self.data_frame)," tracks")

    def __transform_data(self):
        self.data_frame["InvR"] = self.data_frame["trk_pt"].apply(util.pttoR)
        self.data_frame["TanL"] = self.data_frame["trk_eta"].apply(util.tanL)
        self.data_frame = util.predhitpattern(self.data_frame)
        self.config_dict["datatransformed"] = True
        if self.verbose == 1 : print("======Transfromed======")

    def __bit_data(self,normalise : bool = False):
  
      self.data_frame.loc[:,"bit_bendchi2"] = self.data_frame["trk_bendchi2"].apply(np.digitize,bins=self.trackword_config["Bendchi2"]["bins"])
      self.data_frame.loc[:,"bit_chi2rphi"] = self.data_frame["trk_chi2rphi"].apply(np.digitize,bins=self.trackword_config["Chi2rphi"]["bins"])
      self.data_frame.loc[:,"bit_chi2rz"]   = self.data_frame["trk_chi2rz"].apply(np.digitize,bins=self.trackword_config["Chi2rz"]["bins"])
      self.data_frame.loc[:,"bit_phi"]      = self.data_frame["trk_phi"].apply(np.digitize,bins=self.trackword_config["Phi"]["split"])
      self.data_frame["bit_chi2"]           = self.data_frame["bit_chi2rphi"]+self.data_frame["bit_chi2rz"]
      self.data_frame.loc[:,"bit_TanL"]     = self.data_frame["TanL"].apply(util.splitter,part_len=self.trackword_config["TanL"]['split'])
      self.data_frame.loc[:,"bit_z0"]       = self.data_frame["trk_z0"].apply(util.splitter,part_len=self.trackword_config["Z0"]['split'])
      self.data_frame.loc[:,"bit_d0"]       = self.data_frame["trk_d0"].apply(util.splitter,part_len=self.trackword_config["D0"]['split'])
      #self.data_frame["nlay_miss"]          = self.data_frame["trk_hitpattern"].apply(util.set_nlaymissinterior)
      self.data_frame.loc[:,"bit_InvR"]     = self.data_frame["InvR"].apply(util.splitter,part_len=self.trackword_config["InvR"]['split'])  

      self.config_dict["databitted"] = True
      self.config_dict["datanormalised"] = normalise 

      if normalise:
        self.data_frame[["bit_bendchi2","bit_chi2rphi","bit_chi2rz",
                          "bit_InvR","bit_TanL","bit_z0"]] = self.data_frame[["bit_bendchi2","bit_chi2rphi","bit_chi2rz",
                                                                              "bit_InvR","bit_TanL","bit_z0"]]/2**7
        self.data_frame["bit_z0"] = self.data_frame["bit_z0"]/2**4
        self.data_frame["bit_TanL"] = self.data_frame["bit_TanL"]/2**5

        predicted_feature_list = ["pred_nstub","pred_layer1","pred_layer2","pred_layer3",
                                      "pred_layer4","pred_layer5","pred_layer6","pred_disk1",
                                      "pred_disk2","pred_disk3","pred_disk4","pred_disk5",
                                      "pred_dtot","pred_ltot"]
        self.data_frame[predicted_feature_list] = self.data_frame[predicted_feature_list]*2**7  
        if self.verbose == 1 : print("=======Normalised======")
      if self.verbose == 1 : print  ("=====Bit Formatted=====")
      
    def __fake_balance_data(self,fake_to_genuine_ratio : float = 1):

        genuine = util.genuineTracks(self.data_frame)
        numgenuine = len(genuine)

        fake = util.fakeTracks(self.data_frame)
        numfake = len(fake)

        self.config_dict["NumTrainTracks"] = numfake + numgenuine
        self.config_dict["NumTrainFakes"]  = numfake
        self.config_dict["fakebalanced"] = True

        fraction = numfake/numgenuine*fake_to_genuine_ratio

        genuine = genuine.sample(frac=fraction, replace=True, random_state=self.random_state)


        self.data_frame = pd.concat([fake,genuine],ignore_index=True)

        del [fake,genuine]

        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)
        if self.verbose == 1 : print("===Fake Balanced===")

    def __particle_balance_data(self,e_multiplier : float = 0.45,h_multiplier : float = 0.45,f_multiplier : float = 1 ,m_mulitpler : float = 0.1):
        electrons = self.data_frame[(self.data_frame["trk_matchtp_pdgid"] == -11) | (self.data_frame["trk_matchtp_pdgid"] == 11)]
        muons = self.data_frame[(self.data_frame["trk_matchtp_pdgid"] == -13) | (self.data_frame["trk_matchtp_pdgid"] == 13)]
        hadrons = self.data_frame[(self.data_frame["trk_matchtp_pdgid"] != -13) & (self.data_frame["trk_matchtp_pdgid"] != 13) & (self.data_frame["trk_matchtp_pdgid"] != 11) & (self.data_frame["trk_matchtp_pdgid"] != -11)]
        fakes = self.data_frame[self.data_frame["trk_fake"] == 0]

        efraction = e_multiplier * len(muons)/(len(electrons) * m_mulitpler )
        hfraction = h_multiplier * len(muons)/(len(hadrons) * m_mulitpler )
        ffraction = f_multiplier * len(muons)/(len(fakes) * m_mulitpler )

        electrons = electrons.sample(frac=efraction,replace=True,random_state=self.random_state)
        hadrons = hadrons.sample(frac=hfraction,replace=True,random_state=self.random_state)
        fakes = fakes.sample(frac=ffraction,replace=True,random_state=self.random_state)

        self.config_dict["NumTrainFakes"]  = len(fakes)
        self.config_dict["NumTrainElectrons"]  = len(electrons)
        self.config_dict["NumTrainMuons"]  = len(muons)
        self.config_dict["NumTrainHadrons"]  = len(hadrons)
        self.config_dict["NumTrainTracks"] = self.config_dict["NumTrainFakes"] + self.config_dict["NumTrainElectrons"] +  self.config_dict["NumTrainMuons"] +self.config_dict["NumTrainHadrons"]
        self.config_dict["particlebalanced"] = True

        self.data_frame = pd.concat([electrons,muons,hadrons,fakes],ignore_index=True)

        del [electrons,muons,hadrons,fakes]
        gc.collect()
        
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)
        if self.verbose == 1 : print("===Particle Balanced===")

    def __pt_balance_data(self,pt_bins = [2,10,100]):

        df_pt_bins = []
        pt_bin_widths = []

        pt_bins = [util.splitter(util.pttoR(pt),part_len=self.trackword_config["InvR"]['split']) for pt in pt_bins]
        for i in range(len(pt_bins)-1):
            temp_df = self.data_frame[(self.data_frame["bit_InvR"] <= pt_bins[i]) & (self.data_frame["bit_InvR"] > pt_bins[i+1])]
            pt_bin_widths.append(len(temp_df))

        fractions = [min(pt_bin_widths)/pt for pt in pt_bin_widths]


        for i in range(len(pt_bins)-1):
            temp_df = self.data_frame[(self.data_frame["bit_InvR"] <= pt_bins[i]) & (self.data_frame["bit_InvR"] > pt_bins[i+1])]
            temp_df = temp_df.sample(frac=fractions[i],replace=True,random_state=self.random_state)
            df_pt_bins.append(temp_df)

        self.data_frame = pd.concat(df_pt_bins,ignore_index=True)

        del [df_pt_bins,temp_df]
        gc.collect()
        if self.verbose == 1 : print("===Pt Balanced===")

    def generate_test_train(self):

      self.__transform_data()
      self.__bit_data()

      self.training_features.append("trk_matchtp_pdgid")

      X = self.data_frame[self.training_features]
      del self.training_features[-1]

      y = self.data_frame["trk_fake"]
      # Only want to particle balance the training data so we perform train test split, then merge train back together and then particle balance
      X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = self.test_size,random_state=self.random_state) 

      self.data_frame = X_train.join(y_train)

      if self.balance_type == "particle":
        self.__particle_balance_data()
      elif self.balance_type == "fake":
        self.__fake_balance_data()
      elif self.balance_type == "pt":
        self.__pt_balance_data()
      else:
          warnings.warn("The Balance type "+self.balance_type+" is unsupported. No balancing will be performed. Choose from particle, fake or pt. ",stacklevel=3)

      self.X_train = self.data_frame[self.training_features]

      temp_y_train = np.array(self.data_frame["trk_fake"].values.tolist())
      self.y_train["label"] = np.where(temp_y_train>0,1,temp_y_train).tolist()

      self.X_test = X_test[self.training_features]

      temp_y_test = np.array(y_test.values.tolist())
      self.y_test["label"] = np.where(temp_y_test>0,1,temp_y_test).tolist()
      self.config_dict["NumTestTracks"] = len(self.y_test)
      self.data_frame = None
      del[X,y]
      gc.collect()
      self.config_dict["testandtrain"] = True
      if self.verbose == 1 : print("===Train Test Split====")

    def save_h5(self,filepath):
        Path(filepath).mkdir(parents=True, exist_ok=True)

        store = pd.HDFStore(filepath+'full_Dataset.h5')
        store['df'] = self.data_frame  # save it
        self.data_frame = None
        store.close()

        self.config_dict["h5filepath"] = filepath
        self.config_dict["save_timestamp"] = datetime.datetime.now().strftime("%H:%M %d/%m/%y")
        with open(filepath+'config_dict.json', 'w') as f:
          json.dump(self.config_dict, f, indent=4)
        if self.verbose == 1 : print("===Full Data Saved===")

    def load_h5(self,filepath):
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
                self.config_dict =  json.load(f)
            self.config_dict["loaded_timestamp"] =  datetime.datetime.now().strftime("%H:%M %d/%m/%y")
            self.name = self.config_dict["name"]
        else:
            print("No Config Dict")

    def save_test_train_h5(self,filepath):
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
      self.config_dict["save_timestamp"] = datetime.datetime.now().strftime("%H:%M %d/%m/%y")
      with open(filepath+'config_dict.json', 'w') as f:
        json.dump(self.config_dict, f, indent=4)

      if self.verbose == 1 : print("===Train Test Saved====")

    def load_test_train_h5(self,filepath):
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
            self.config_dict =  json.load(f)
        self.config_dict["loaded_timestamp"] =  datetime.datetime.now().strftime("%H:%M %d/%m/%y")
        self.name = self.config_dict["name"]
      else:
          print("No configuration dictionary json file")
    
    def __str__(self):
        print("=============================")
        print("Dataset Name: ",self.config_dict["name"])
        print("With random seed: ",self.config_dict["randomState"])
        print("Loaded From ROOT at: ",self.config_dict["rootLoaded"])
        print("from: ",self.config_dict["Rootfilepath"])
        print("Original Number of Events: ",self.config_dict["NumEvents"])
        print("Transformed          |",self.config_dict["datatransformed"])
        print("Bit formatted        |",self.config_dict["databitted"])
        print("Normalised           |",self.config_dict["datanormalised"])
        print("Balance type         |",self.config_dict["balanced"])
        print("Test and Train Split |",self.config_dict["testandtrain"])
        print("=============================")

        if (self.config_dict["balanced"] == "particle"): 
            print("Test and Train Filepath: ",self.config_dict["testandtrainfilepath"])
            print("Saved at: ",self.config_dict["save_timestamp"])
            print("Loaded at: ",self.config_dict["loaded_timestamp"])
            print("Total Tracks    |",self.config_dict["NumTracks"])
            print("Train Tracks    |",self.config_dict["NumTrainTracks"])
            print("Train Fakes     |",self.config_dict["NumTrainFakes"])
            print("Train Electrons |",self.config_dict["NumTrainElectrons"])
            print("Train Muons     |",self.config_dict["NumTrainMuons"])
            print("Train Hadrons   |",self.config_dict["NumTrainHadrons"])
            print("Total Test      |",self.config_dict["NumTestTracks"])
            print("Test Fraction   |",self.config_dict["testsize"])
            print("Training Features: ",self.config_dict["trainingFeatures"])

        else:
            print("h5 Filepath: ",self.config_dict["h5filepath"])
            print("Saved at: ",self.config_dict["save_timestamp"])
            print("Loaded at: ",self.config_dict["loaded_timestamp"])
        return "============================="

    def write_hls_file(self,filepath,num_events=10):
        self.__transform_data()
        self.__bit_data()
        Path(filepath).mkdir(parents=True, exist_ok=True)
        with open(filepath+"input_hls.txt",'w') as f:
            for i in range(num_events):
                f.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} 0 0 \n".format(int(self.data_frame.iloc[i]["bit_InvR"]),
                                                                          int(self.data_frame.iloc[i]["bit_phi"]),
                                                                          int(self.data_frame.iloc[i]["bit_TanL"]),
                                                                          int(self.data_frame.iloc[i]["bit_z0"]),
                                                                          int(self.data_frame.iloc[i]["bit_d0"]),
                                                                          int(self.data_frame.iloc[i]["bit_chi2rphi"]),
                                                                          int(self.data_frame.iloc[i]["bit_chi2rz"]),
                                                                          int(self.data_frame.iloc[i]["bit_bendchi2"]),
                                                                          int(self.data_frame.iloc[i]["trk_hitpattern"]),
                                                                        ))
        f.close()

class EventDataSet(DataSet):
    def __init__(self,name):
        super().__init__(name)

        self.feature_list = ['trk_fake','trk_pt','trk_z0',"trk_eta"]
        self.training_features = ['trk_fake','trk_pt','trk_z0',"trk_eta","TanL","InvR"]
        self.pv_list = []

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.config_dict = {"name":self.name,
                            "rootLoaded":None,
                            "Rootfilepath":"",
                            "datatransformed":False,
                            "PV_found":False,
                            "testandtrain":False,
                            "testandtrainfilepath":"",
                            "h5filepath":"",
                            "NumEvents":0,
                            "trainingFeatures":self.training_features,
                            "randomState":self.random_state,
                            "testsize":self.test_size,
                            "save_timestamp":None,
                            "loaded_timestamp":None
                           }

    def __transform_data(self,event):
        event["InvR"] = util.pttoR(event["trk_pt"])
        event["TanL"] = util.tanL(event["trk_eta"])
        self.config_dict["datatransformed"] = True
        if self.verbose == 1 : print("======Transfromed======")
        return event

    def load_data_from_root(self,filepath,numEvents):

        data = uproot.open(filepath+".root:L1TrackNtuple;1/eventTree;1",numEvents=numEvents).arrays(self.feature_list)

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



        del [y,data]
        gc.collect()
        self.data_frame = events
        self.config_dict["NumEvents"] = len(events)

    def generate_test_train(self):

      # Only want to particle balance the training data so we perform train test split, then merge train back together and then particle balance
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_frame,self.pv_list,test_size = self.test_size,random_state=self.random_state) 

      self.config_dict["NumTestEvents"] = len(self.y_test)
      self.config_dict["NumTrainEvents"] = len(self.y_train)
      self.data_frame = None
      self.pv_list = None

      self.config_dict["testandtrain"] = True
      if self.verbose == 1 : print("===Train Test Split====")
    
    def save_test_train_h5(self,filepath):
      Path(filepath).mkdir(parents=True, exist_ok=True)

      with open(filepath+'X_train.h5', 'wb') as f_xtrain:
        pickle.dump(self.X_train, f_xtrain, protocol=pickle.HIGHEST_PROTOCOL)

      with open(filepath+'X_test.h5', 'wb') as f_xtest:
        pickle.dump(self.X_test, f_xtest, protocol=pickle.HIGHEST_PROTOCOL)

      with open(filepath+'y_train.h5', 'wb') as f_ytrain:
        pickle.dump(self.y_train, f_ytrain, protocol=pickle.HIGHEST_PROTOCOL)

      with open(filepath+'y_test.h5', 'wb') as f_ytest:
        pickle.dump(self.y_test, f_ytest, protocol=pickle.HIGHEST_PROTOCOL)

      self.X_train = None
      self.X_test = None
      self.y_train = None
      self.y_test = None

      self.config_dict["testandtrainfilepath"] = filepath
      self.config_dict["save_timestamp"] = datetime.datetime.now().strftime("%H:%M %d/%m/%y")
      with open(filepath+'config_dict.json', 'w') as f:
        json.dump(self.config_dict, f, indent=4)

      if self.verbose == 1 : print("===Train Test Saved====")

    def load_test_train_h5(self,filepath):
      X_train_file = Path(filepath+'X_train.h5')
      if X_train_file.is_file():
        with open(filepath+'X_train.h5', 'rb') as f_xtrain:
            self.X_train = pickle.load(f_xtrain)
      else:
          print("No X Train h5 File")

      X_test_file = Path(filepath+'X_test.h5')
      if X_test_file .is_file():
          with open(filepath+'X_test.h5', 'rb') as f_xtest:
            self.X_test = pickle.load(f_xtest)
      else:
          print("No X Test h5 File")

      y_train_file = Path(filepath+'y_train.h5')
      if y_train_file.is_file():
          with open(filepath+'y_train.h5', 'rb') as f_ytrain:
            self.y_train = pickle.load(f_ytrain)
      else:
          print("No y train h5 file")

      y_test_file = Path(filepath+'y_test.h5')
      if y_test_file.is_file():
        with open(filepath+'y_test.h5', 'rb') as f_ytest:
            self.y_test= pickle.load(f_ytest)
      else:
          print("No y test h5 file")

      config_dict_file = Path(filepath+'config_dict.json')
      if config_dict_file.is_file():
        with open(filepath+'config_dict.json', 'r') as f:
            self.config_dict =  json.load(f)
        self.config_dict["loaded_timestamp"] =  datetime.datetime.now().strftime("%H:%M %d/%m/%y")
        self.name = self.config_dict["name"]
      else:
          print("No configuration dictionary json file")
    
    def find_PV(self):
        self.pv_list = []
        for event in self.data_frame:
            pvtrks = (event[event["trk_fake"]==1])
            self.pv_list.append(np.average(pvtrks["trk_z0"],weights=pvtrks["trk_pt"]))
        self.config_dict["PV_found"] = True

    def find_fast_hist(self):
        self.z0List = []
        halfBinWidth = 0.5*30./256.
        for event in self.data_frame:
            hist,bin_edges = np.histogram(event["trk_z0"],256,range=(-15,15),weights=event["trk_pt"])
            hist = np.convolve(hist,[1,1,1],mode='same')
            z0Index= np.argmax(hist)
            z0 = -15.+30.*z0Index/256.+halfBinWidth
            self.z0List.append([z0])

    def __str__(self):
        print("=============================")
        print("Dataset Name: ",self.config_dict["name"])
        print("With random seed: ",self.config_dict["randomState"])
        print("Loaded From ROOT at: ",self.config_dict["rootLoaded"])
        print("from: ",self.config_dict["Rootfilepath"])
        print("Original Number of Events: ",self.config_dict["NumEvents"])
        print("Transformed          |",self.config_dict["datatransformed"])
        print("PVs Found            |",self.config_dict["PV_found"])
        print("Test and Train Split |",self.config_dict["testandtrain"])
        print("=============================")
        if self.config_dict["testandtrain"]:
            print("Test and Train Filepath: ",self.config_dict["testandtrainfilepath"])
            print("Saved at: ",self.config_dict["save_timestamp"])
            print("Loaded at: ",self.config_dict["loaded_timestamp"])
            print("Test Tracks     |",self.config_dict["NumTrainEvents"])
            print("Test Events     |",self.config_dict["NumTestEvents"])
            print("Test Fraction   |",self.config_dict["testsize"])
            print("Training Features: ",self.config_dict["trainingFeatures"])
        return "============================="
