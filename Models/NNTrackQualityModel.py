from comet_ml import Optimizer
from Datasets.Dataset import DataSet
import numpy as np
from Models.TrackQualityModel import TrackClassifierModel

import keras
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import Input
import models
from callbacks import all_callbacks
from tensorflow.keras import callbacks

from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning


class KerasClassifierModel(TrackClassifierModel):
    def __init__(self):
        super().__init__()

        self.min_child_weight =  {"min":0,"max":10, "value":8.582127}
        self.alpha            =  {"min":0,"max":1,  "value":0.54012275}
        self.early_stopping   =  {"min":1,"max":20, "value":5}
        self.learning_rate    =  {"min":0,"max":1,   "value":0.6456304}
        self.n_estimators     =  100
        self.subsample        =  {"min":0,"max":0.99,"value":0.343124}
        self.max_depth        =  {"min":1,"max":3  ,"value":3  }
        self.gamma            =  {"min":0,"max":0.99,"value":0.314177}
        self.rate_drop        =  {"min":0,"max":1,"value":0.788588}
        self.skip_drop        =  {"min":0,"max":1,"value":0.147907}

        self.dtest = None
        self.dtrain = None

        self.comet_project_name = "Keras_test"

        self.param = {"max_depth":self.max_depth["value"],
                      "learning_rate":self.learning_rate["value"],
                      "min_child_weight":self.min_child_weight["value"],
                      "gamma":self.gamma["value"],
                      'eval_metric':'auc',
                      "subsample":self.subsample["value"],
                      "reg_alpha":self.alpha["value"] ,
                      "objective":'binary:logistic',
                      "nthread":8,
                      "booster":"dart",
                      "rate_drop":self.rate_drop["value"] ,
                      "skip_drop":self.skip_drop["value"]}
        self.num_rounds = self.n_estimators
        self.early_stopping_rounds = self.early_stopping["value"]


        self.model = None

    def load_data(self,filepath):
        self.DataSet = DataSet.fromTrainTest(filepath)
        mult = self.DataSet.trackword_config["TargetPrecision"]["full"] - \
                self.DataSet.trackword_config["TargetPrecision"]["int"]
        self.DataSet.X_train["b_trk_inv2R"] = self.DataSet.X_train["b_trk_inv2R"] * \
                (2**(mult-self.DataSet.trackword_config["InvR"]["nbits"]))
        self.DataSet.X_train["b_trk_cot"] = self.DataSet.X_train["b_trk_cot"] * \
                (2**(mult-self.DataSet.trackword_config["Cot"]["nbits"]))
        self.DataSet.X_train["b_trk_zT"] = self.DataSet.X_train["b_trk_zT"] * \
                (2**(mult-self.DataSet.trackword_config["ZT"]["nbits"]))
        self.DataSet.X_train["b_trk_phiT"] = self.DataSet.X_train["b_trk_phiT"] * \
                (2**(mult-self.DataSet.trackword_config["PhiT"]["nbits"]))
        # ==============================================================================
        self.DataSet.X_test["b_trk_inv2R"] = self.DataSet.X_test["b_trk_inv2R"] * \
                (2**(mult-self.DataSet.trackword_config["InvR"]["nbits"]))
        self.DataSet.X_test["b_trk_cot"] = self.DataSet.X_test["b_trk_cot"] * \
                (2**(mult-self.DataSet.trackword_config["Cot"]["nbits"]))
        self.DataSet.X_test["b_trk_zT"] = self.DataSet.X_test["b_trk_zT"] * \
                (2**(mult-self.DataSet.trackword_config["ZT"]["nbits"]))
        self.DataSet.X_test["b_trk_phiT"] = self.DataSet.X_test["b_trk_phiT"] * \
                (2**(mult-self.DataSet.trackword_config["PhiT"]["nbits"]))

        for k in range(4):
            self.DataSet.X_test["b_stub_r_" + str(k)] = self.DataSet.X_test["b_stub_r_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["r"]["nbits"]))
            self.DataSet.X_test["b_stub_phi_" + str(k)] = self.DataSet.X_test["b_stub_phi_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["phi"]["nbits"]))
            self.DataSet.X_test["b_stub_z_" + str(k)] = self.DataSet.X_test["b_stub_z_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["z"]["nbits"]))
            self.DataSet.X_test["b_stub_dPhi_" + str(k)] = self.DataSet.X_test["b_stub_dPhi_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["dPhi"]["nbits"]))
            self.DataSet.X_test["b_stub_dZ_" + str(k)] = self.DataSet.X_test["b_stub_dZ_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["dZ"]["nbits"]))
            # ==============================================================================
            self.DataSet.X_train["b_stub_r_" + str(k)] = self.DataSet.X_train["b_stub_r_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["r"]["nbits"]))
            self.DataSet.X_train["b_stub_phi_" + str(k)] = self.DataSet.X_train["b_stub_phi_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["phi"]["nbits"]))
            self.DataSet.X_train["b_stub_z_" + str(k)] = self.DataSet.X_train["b_stub_z_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["z"]["nbits"]))
            self.DataSet.X_train["b_stub_dPhi_" + str(k)] = self.DataSet.X_train["b_stub_dPhi_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["dPhi"]["nbits"]))
            self.DataSet.X_train["b_stub_dZ_" + str(k)] = self.DataSet.X_train["b_stub_dZ_" + str(k)] * \
                (2**(mult-self.DataSet.trackword_config["dZ"]["nbits"]))

    def train(self):
        
        cv = xgb.cv(self.param, self.dtrain, self.num_rounds, nfold=5,
               metrics={'auc'}, seed=4,
               callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True),xgb.callback.EarlyStopping(self.early_stopping_rounds)])

        self.boost_rounds = cv['test-auc-mean'].argmax()
        self.model = xgb.train(self.param,self.dtrain,num_boost_round=self.boost_rounds)

    def test(self):
        
        self.y_predict_proba = self.model.predict(self.dtest)
        self.y_predict = np.heaviside(self.y_predict_proba -0.5,0)

    def optimise(self):
        config = {
                    # We pick the Bayes algorithm:
                    "algorithm": "bayes",

                    # Declare your hyperparameters in the Vizier-inspired format:
                    "parameters": {
                        "max_depth":        {"type": "integer", "min": self.max_depth["min"], 
                                             "max": self.max_depth["max"],        "scalingType": "uniform"},
                        "learning_rate":    {"type": "float", "min": self.learning_rate["min"], 
                                             "max": self.learning_rate["max"],    "scalingType": "uniform"},
                        "gamma":            {"type": "float", "min": self.gamma["min"], 
                                             "max": self.gamma["max"] ,           "scalingType": "uniform"},
                        "subsample":        {"type": "float", "min": self.subsample["min"], 
                                             "max": self.subsample["max"],        "scalingType": "uniform"},
                        "min_child_weight": {"type": "float", "min": self.min_child_weight["min"], 
                                             "max": self.min_child_weight["max"], "scalingType": "uniform"},
                        "reg_alpha":        {"type": "float", "min": self.alpha["min"], 
                                             "max": self.alpha["max"],            "scalingType": "uniform"},
                        "early_stopping":   {"type": "integer", "min": self.early_stopping["min"], 
                                             "max": self.early_stopping["max"],     "scalingType": "uniform"},
                        "rate_drop":        {"type": "float", "min": self.rate_drop["min"], 
                                             "max": self.rate_drop["max"],     "scalingType": "uniform"},
                        "skip_drop":        {"type": "float", "min": self.skip_drop["min"], 
                                             "max": self.skip_drop["max"],     "scalingType": "uniform"}
                    },

                    # Declare what we will be optimizing, and how:
                    "spec": {
                    "metric": "ROC",
                        "objective": "maximize",
                    },
                }
        opt = Optimizer(config, api_key=self.comet_api_key, project_name=self.comet_project_name,auto_metric_logging=True)

        for experiment in opt.get_experiments():
            self.param["learning_rate"] = experiment.get_parameter("learning_rate")
            self.num_rounds = self.n_estimators
            self.param["subsample"]= experiment.get_parameter("subsample")
            self.param["max_depth"] = experiment.get_parameter("max_depth")
            self.param["gamma"] = experiment.get_parameter("gamma")
            self.param["reg_alpha"] =experiment.get_parameter("reg_alpha")
            self.param["min_child_weight"] = experiment.get_parameter("min_child_weight")
            self.param["rate_drop"] = experiment.get_parameter("rate_drop")
            self.param["skip_drop"] = experiment.get_parameter("skip_drop")
            self.early_stopping_rounds =  experiment.get_parameter("early_stopping")

            self.train()
            self.test()
            auc,binary_accuracy = self.evaluate()

            experiment.log_metric("ROC",auc)
            experiment.log_metric("Binary_Accuracy",binary_accuracy)
            experiment.log_metric("Best Boost Round",self.boost_rounds)
            experiment.log_metric("score",(auc/0.5-1)+(1-self.boost_rounds/self.n_estimators ))

    def plot_model(self):
        xgb.plot_importance(self.model)

    def synth_model(self,sim : bool = True,hdl : bool = True,hls : bool = True,test_events : int = 1000):
        import shutil

        if sim:
            if Path("simdir").is_dir():
              shutil.rmtree("simdir")

            from scipy.special import expit
            from conifer import conifer
            simcfg = conifer.backends.vhdl.auto_config()
            simcfg['Precision'] = 'ap_fixed<13,6>'
            # Set the output directory to something unique
            simcfg['OutputDir'] = "simdir/"
            simcfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
            simcfg["ClockPeriod"] = 2.78

            self.simmodel = conifer.model(self.model, conifer.converters.xgboost, conifer.backends.vhdl, simcfg)
            self.simmodel.compile()

            # Run HLS C Simulation and get the output
            

            if test_events > 0:
                length = test_events
            
                temp_decision = self.simmodel.decision_function(self.DataSet.X_test[0:length])
                self.synth_y_predict_proba = expit(temp_decision)
                temp_array = np.empty_like(temp_decision)
                temp_array[self.synth_y_predict_proba > 0.5] = 1
                temp_array[self.synth_y_predict_proba <= 0.5] = 0
                self.synth_y_predict = temp_array
                

                print("AUC ROC sim: {}".format(metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.synth_y_predict_proba)))
                print("AUC ROC sklearn: {}".format(metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.y_predict_proba[0:len(self.synth_y_predict_proba)])))
                print("Accuracy sim: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.synth_y_predict)))
                print("Accuracy sklearn: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.y_predict[0:len(self.synth_y_predict_proba)])))


        if hdl:
            if Path("hdldir").is_dir():
                shutil.rmtree("hdldir")

            hdlcfg = conifer.backends.vhdl.auto_config()
            hdlcfg['Precision'] = 'ap_fixed<13,6>'
            # Set the output directory to something unique
            hdlcfg['OutputDir'] = "hdldir/"
            hdlcfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
            hdlcfg["ClockPeriod"] = 2.78
            hdlmodel = conifer.model(self.model, conifer.converters.xgboost, conifer.backends.vhdl, hdlcfg)
            hdlmodel.write()
            hdlmodel.build()
        if hls:
            if Path("hlsdir").is_dir():
                shutil.rmtree("hlsdir")

            hlscfg = conifer.backends.vivadohls.auto_config()
            hlscfg['Precision'] = 'ap_fixed<13,6>'
            # Set the output directory to something unique
            hlscfg['OutputDir'] = "hlsdir/"
            hlscfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
            hlscfg["ClockPeriod"] = 2.78

            # Create and compile the model
            hlsmodel = conifer.model(self.model, conifer.converters.xgboost, conifer.backends.vivadohls, hlscfg)
            hlsmodel.write()
            hlsmodel.build()

