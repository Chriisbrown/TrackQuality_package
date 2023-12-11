from comet_ml import Optimizer
from Datasets.Dataset import DataSet
from sklearn import metrics
import numpy as np
import joblib
import os
import xgboost as xgb 
import pickle
# import tensorflow_decision_forests as tfdf
# import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier 
from Models.TrackQualityModel import TrackClassifierModel
from pathlib import Path

class GBDTClassifierModel(TrackClassifierModel):
    def __init__(self,name):
        super().__init__(name)

        self.learning_rate =  {"min":0,"max":1,   "value":0.1}
        self.n_estimators  =  {"min":0,"max":150,"value":100}
        self.subsample     =  {"min":0,"max":0.99,"value":0.5}
        self.max_depth     =  {"min":1,"max":5  ,"value":3  }
        self.gamma         =  {"min":0,"max":0.99,"value":0.0}

        self.output_bins =  np.array([0,0.125,0.250,0.375,0.5,0.625,0.750,0.825,1.00]) #n

        self.model = None 

    def train(self):
        self.model = self.model.fit(self.DataSet.X_train[self.training_features].to_numpy(),np.ravel(self.DataSet.y_train),verbose=1)

    def retrain(self):
        self.model = self.model.fit(self.DataSet.X_train[self.training_features].to_numpy(),np.ravel(self.DataSet.y_train),xgb_model=self.model,verbose=1)

    def test(self):
        self.y_predict = self.model.predict(self.DataSet.X_test[self.training_features].to_numpy())
        self.y_predict_proba = self.model.predict_proba(self.DataSet.X_test[self.training_features].to_numpy())[:,1]
        self.y_predict_binned = np.digitize(self.y_predict_proba,bins=self.output_bins)
        self.y_predict_binned = (self.y_predict_binned - 1) / 8

    def save_model(self,filepath):
        joblib.dump(self.model,filepath + ".pkl")

    def optimise(self):
        pass

    def synth_model(self):
        pass

class SklearnClassifierModel(GBDTClassifierModel):
    def __init__(self):
        
        super().__init__()

        self.backend = "sklearn"

        self.model = GradientBoostingClassifier(loss="deviance",
                                                learning_rate = self.learning_rate["value"],
                                                n_estimators  = self.n_estimators["value"],
                                                subsample     = self.subsample["value"],
                                                max_depth     = self.max_depth["value"],
                                                ccp_alpha     = self.gamma["value"],
                                                verbose       = 2,
                                                )

    def optimise(self):
        config = {
                    # We pick the Bayes algorithm:
                    "algorithm": "bayes",

                    # Declare your hyperparameters in the Vizier-inspired format:
                    "parameters": {
                        "n_estimators":  {"type": "int", "min": self.n_estimators["min"], 
                                          "max": self.n_estimators["max"],  "scalingType": "uniform"},
                        "max_depth":     {"type": "int", "min": self.max_depth["min"], 
                                          "max": self.max_depth["max"],     "scalingType": "uniform"},
                        "learning_rate": {"type": "float", "min": self.learning_rate["min"], 
                                          "max": self.learning_rate["max"], "scalingType": "uniform"},
                        "gamma":         {"type": "float", "min": self.gamma["min"], 
                                          "max": self.gamma["max"] ,        "scalingType": "uniform"},
                        "subsample":     {"type": "float", "min": self.subsample["min"], 
                                          "max": self.subsample["max"],     "scalingType": "uniform"},
                    },

                    # Declare what we will be optimizing, and how:
                    "spec": {
                    "metric": "ROC",
                        "objective": "maximize",
                    },
                }
        opt = Optimizer(config, api_key=self.comet_api_key, project_name=self.comet_project_name,auto_metric_logging=True)

        for experiment in opt.get_experiments():
            self.model.learning_rate = experiment.get_parameter("learning_rate")
            self.model.n_estimators = experiment.get_parameter("n_estimators")
            self.model.subsample = experiment.get_parameter("subsample")
            self.model.max_depth = experiment.get_parameter("max_depth")
            self.model.ccp_alpha = experiment.get_parameter("gamma")

            self.train()
            self.test()
            auc,binary_accuracy = self.evaluate()

            experiment.log_metric("ROC",auc)
            experiment.log_metric("Binary_Accuracy",binary_accuracy)

class XGBoostClassifierModel(GBDTClassifierModel):
    def __init__(self,name):
        super().__init__(name)

        self.backend = "xgboost"

        self.min_child_weight =  {"min":0,"max":10, "value":1.3690275705621135}
        self.alpha            =  {"min":0,"max":1,  "value":0.9307933560230425}
        self.early_stopping   =  {"min":1,"max":20, "value":5}
        self.learning_rate    =  {"min":0,"max":1,   "value":0.3245287291246959}
        self.n_estimators     =  {"min":0,"max":1000,   "value":60}
        self.subsample        =  {"min":0,"max":0.99,"value":0.2459092973919883	}
        self.max_depth        =  {"min":1,"max":10  ,"value":3 }
        self.gamma            =  {"min":0,"max":0.99,"value":0.0	}
        self.rate_drop        =  {"min":0,"max":1,"value":0.788588}
        self.skip_drop        =  {"min":0,"max":1,"value":0.147907}

        self.model = xgb.XGBClassifier(n_estimators      = self.n_estimators["value"],
                                       max_depth         = self.max_depth["value"],
                                       learning_rate     = self.learning_rate["value"],
                                       gamma             = self.gamma["value"],
                                       min_child_weight  = self.min_child_weight["value"],
                                       subsample         = self.subsample["value"],
                                       reg_alpha         = self.alpha["value"] ,
                                       objective         = 'binary:logistic',
                                       #booster           = "dart",
                                       #rate_drop         = self.rate_drop["value"] ,
                                       #skip_drop         = self.skip_drop["value"] ,
                                       tree_method       = 'exact',
                                       use_label_encoder = False ,
                                       n_jobs            = 8)

    def optimise(self):
        config = {
                    # We pick the Bayes algorithm:
                    "algorithm": "bayes",

                    # Declare your hyperparameters in the Vizier-inspired format:
                    "parameters": {
                        "n_estimators":     {"type": "integer", "min": self.n_estimators["min"], 
                                             "max": self.n_estimators["max"],     "scalingType": "uniform"},
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
                    },

                    # Declare what we will be optimizing, and how:
                    "spec": {
                    "metric": "ROC",
                        "objective": "maximize",
                    },
                }
        opt = Optimizer(config, api_key=self.comet_api_key, project_name=self.comet_project_name,auto_metric_logging=True)

        for experiment in opt.get_experiments():
            self.model.learning_rate = experiment.get_parameter("learning_rate")
            self.model.n_estimators = experiment.get_parameter("n_estimators")
            self.model.subsample = experiment.get_parameter("subsample")
            self.model.max_depth = experiment.get_parameter("max_depth")
            self.model.gamma = experiment.get_parameter("gamma")
            self.model.reg_alpha =experiment.get_parameter("reg_alpha")
            self.model.min_child_weight = experiment.get_parameter("min_child_weight")

            self.train()
            self.test()
            auc,binary_accuracy = self.evaluate()

            experiment.log_metric("ROC",auc)
            experiment.log_metric("Binary_Accuracy",binary_accuracy)

    def save_model(self,filepath,name):
        if not Path(filepath).is_dir():
            os.system("mkdir -p " + filepath)
        self.model.save_model(filepath + name + ".json")
        #self.model.dump_model(filepath + name + ".json")
        #pickle.dump(self.model, open(filepath + name + ".pkl", "wb"))
        
    def load_model(self,filepath,name):
        self.model.load_model(filepath + name +".json")
        #self.model = pickle.load(open(filepath + name + ".pkl", "rb"))

class FullXGBoostClassifierModel(XGBoostClassifierModel):
    def __init__(self):
        super().__init__()
        self.backend = "xgboost"

        self.min_child_weight =  {"min":0,"max":10, "value":1.3690275705621135}
        self.alpha            =  {"min":0,"max":1,  "value":0.9307933560230425}
        self.early_stopping   =  {"min":1,"max":20, "value":5}
        self.learning_rate    =  {"min":0,"max":1,   "value":0.3245287291246959}
        self.n_estimators     =  100
        self.subsample        =  {"min":0,"max":0.99,"value":0.2459092973919883}
        self.max_depth        =  {"min":1,"max":3  ,"value":3  }
        self.gamma            =  {"min":0,"max":0.99,"value":30.0}
        self.rate_drop        =  {"min":0,"max":1,"value":0.788588}
        self.skip_drop        =  {"min":0,"max":1,"value":0.147907}

        self.dtest = None
        self.dtrain = None

        self.comet_project_name = "Xgboost_test"

        self.param = {"max_depth":self.max_depth["value"],
                      "learning_rate":self.learning_rate["value"],
                      "min_child_weight":self.min_child_weight["value"],
                      "gamma":self.gamma["value"],
                      'eval_metric':'auc',
                      "subsample":self.subsample["value"],
                      "reg_alpha":self.alpha["value"] ,
                      "objective":'binary:logistic',
                      "nthread":16,
                      "booster":"dart",
                      "rate_drop":self.rate_drop["value"] ,
                      "skip_drop":self.skip_drop["value"]}
        self.num_rounds = self.n_estimators
        self.early_stopping_rounds = self.early_stopping["value"]


        self.model = None

    def load_data(self,filepath):
        self.DataSet = DataSet.fromTrainTest(filepath)
        self.dtrain = xgb.DMatrix(self.DataSet.X_train[self.training_features].to_numpy(),label=np.ravel(self.DataSet.y_train))
        self.dtest = xgb.DMatrix(self.DataSet.X_test[self.training_features].to_numpy(),label=np.ravel(self.DataSet.y_test))

    def load_model(self,filepath):
        self.model = xgb.Booster()
        self.model.load_model(filepath + ".json")

    def train(self):
        
        cv = xgb.cv(self.param, self.dtrain, self.num_rounds, nfold=5,
               metrics={'auc'}, seed=4,
               callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True),xgb.callback.EarlyStopping(self.early_stopping_rounds)])

        self.boost_rounds = cv['test-auc-mean'].argmax()
        self.model = xgb.train(self.param,self.dtrain,num_boost_round=self.boost_rounds)

    def test(self):        
        self.y_predict_proba = self.model.predict(self.dtest,iteration_range=(0, self.model.best_iteration+1))
        self.y_predict = np.heaviside(self.y_predict_proba -0.5,0)
        self.y_predict_binned = np.digitize(self.y_predict_proba,bins=self.output_bins)
        self.y_predict_binned = self.y_predict_binned / 8

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

class TFDFClassifierModel(GBDTClassifierModel):
    def __init__(self):
        super().__init__()

        self.backend = "tfdf"

        self.min_child_weight =  {"min":0,"max":10, "value":1.3690275705621135}
        self.alpha            =  {"min":0,"max":1,  "value":0.9307933560230425}
        self.early_stopping   =  {"min":1,"max":20, "value":5}
        self.learning_rate    =  {"min":0,"max":1,   "value":0.3245287291246959}
        self.n_estimators     =  {"min":0,"max":250,   "value":5}
        self.subsample        =  {"min":0,"max":0.99,"value":0.2459092973919883	}
        self.max_depth        =  {"min":1,"max":5  ,"value":3 }
        self.gamma            =  {"min":0,"max":0.99,"value":0.0	}
        self.rate_drop        =  {"min":0,"max":1,"value":0.788588}
        self.skip_drop        =  {"min":0,"max":1,"value":0.147907}


        self.model = tfdf.keras.GradientBoostedTreesModel(num_trees=self.n_estimators["value"],
                                                          max_depth=self.max_depth["value"],
                                                          verbose = 1)
        
    def save_model(self,filepath):
        self.model.save(filepath,  save_format='tf')

    def load_model(self,filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.model.compile(metrics=["binary_crossentropy"])


    def load_data(self,filepath):
        self.DataSet = DataSet.fromTrainTest(filepath)
        self.DataSet.X_train["label"] = self.DataSet.y_train
        self.DataSet.X_test["label"] = self.DataSet.y_test
        self.train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.DataSet.X_train[self.training_features],"label")
        self.test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.DataSet.X_test[self.training_features],"label")

    def train(self):
        self.model.compile(metrics=["binary_crossentropy"])
        self.model.fit(x=self.train_ds)
        print(self.model.summary())
        print(self.model.make_inspector().variable_importances())

    def test(self):
        #self.model.compile(metrics=["binary_crossentropy"])
        evaluation = self.model.evaluate(self.test_ds, return_dict=True)
        for name, value in evaluation.items():
            print(f"{name}: {value:.4f}")

        self.y_predict_proba = self.model.predict(self.test_ds)
        self.y_predict = np.heaviside(self.y_predict_proba -0.5,0)
        self.y_predict_binned = np.digitize(self.y_predict_proba,bins=self.output_bins)
        self.y_predict_binned = self.y_predict_binned / 8