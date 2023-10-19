from comet_ml import Optimizer
from Dataset import DataSet
from sklearn import metrics
from sklearn.inspection import permutation_importance
import numpy as np
import joblib
from pathlib import Path
import os
import xgboost as xgb 
import pickle
import conifer
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import util
import matplotlib
matplotlib.use('Agg')
import mplhep as hep
        #hep.set_style("CMSTex")

hep.cms.label()
hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

LEGEND_WIDTH = 20
LINEWIDTH = 3
MARKERSIZE = 20

colormap = "jet"

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=LINEWIDTH)              # thickness of axes
plt.rc('xtick', labelsize=SMALL_SIZE+2)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-2)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 10
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 3

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 2

colours=["red","black","blue","orange","purple","goldenrod",'green']
linestyles = ["-","--","dotted",(0, (3, 5, 1, 5)),(0, (3, 5, 1, 5, 1, 5)),(0, (3, 10, 1, 10)),(0, (3, 10, 1, 10, 1, 10))]

from TrackQualityModel import TrackClassifierModel

def SynthBDTmodel(model,backend,cfg,directory,precision,test_events=10000,build=False):
        from scipy.special import expit
        
        cfg['Precision'] = precision
            # Set the output directory to something unique
        cfg['OutputDir'] = directory
        cfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
        cfg["ClockPeriod"] = 2.78

        match backend:
            case 'xgboost':
                converted_model = conifer.converters.convert_from_xgboost(model.model, cfg)
            case 'tfdf':
                converted_model = conifer.converters.convert_from_tf_df(model.inspector, cfg)
        converted_model.compile()
            # Run HLS C Simulation and get the output
        if test_events > 0:
            temp_decision = converted_model.decision_function(model.DataSet.X_test[0:test_events].to_numpy())
            synth_y_predict_proba = expit(temp_decision)
        
        report = None

        if build:
            converted_model.build()
            converted_model.write()
            report = converted_model.read_report()

        return (synth_y_predict_proba,report)
            
def ONNX_convert_model(model,backend,directory,test_events=10000):
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        import onnxruntime as rt

        num_features = len(model.DataSet.config_dict["trainingFeatures"])
        X = model.DataSet.X_test[0:test_events]
        
        # The name of the input is needed in Clasifier_cff as GBDTIdONNXInputName
        initial_type = [('feature_input', FloatTensorType([1, num_features]))]

        match backend:
            case 'xgboost':
                onx = onnxmltools.convert.convert_xgboost(model.model, initial_types=initial_type)

            case 'tfdf':
                onx = onnxmltools.convert.convert_xgboost(model.model, initial_types=initial_type)

        # Save the model
        with open(directory+".onnx", "wb") as f:
          f.write(onx.SerializeToString())

        # This tests the model

        # setup runtime - load the persisted ONNX model
        sess = rt.InferenceSession(directory+".onnx")

        # get model metadata to enable mapping of new input to the runtime model.
        input_name = sess.get_inputs()[0].name
        # This label will access the class probabilities when run in CMSSW, use index 0 for class prediction
        label_name = sess.get_outputs()[1].name


        #print(sess.get_inputs()[0].name)
        # The name of the output is needed in Clasifier_cff as GBDTIdONNXOutputName
        #print(label_name)
        pred_onx = []
        # predict on random input and compare to previous XGBoost model
        for i in range(len(X)):
          pred_onx.append(sess.run([], {input_name: X[i:i+1].to_numpy(dtype=np.float32)})[1][0][1])
        return pred_onx
class GBDTClassifierModel(TrackClassifierModel):
    def __init__(self):
        super().__init__()

        self.learning_rate =  {"min":0,"max":1,   "value":0.1}
        self.n_estimators  =  {"min":0,"max":150,"value":100}
        self.subsample     =  {"min":0,"max":0.99,"value":0.5}
        self.max_depth     =  {"min":1,"max":5  ,"value":3  }
        self.gamma         =  {"min":0,"max":0.99,"value":0.0}

        self.output_bins =  np.array([0,0.125,0.250,0.375,0.5,0.625,0.750,0.825,1.00]) #np.array([0,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.00])#

        self.model = None 

    def train(self):
        self.model = self.model.fit(self.DataSet.X_train.to_numpy(),np.ravel(self.DataSet.y_train),verbose=1)

    def test(self):
        self.y_predict = self.model.predict(self.DataSet.X_test.to_numpy())
        self.y_predict_proba = self.model.predict_proba(self.DataSet.X_test.to_numpy())[:,1]
        self.y_predict_binned = np.digitize(self.y_predict_proba,bins=self.output_bins)
        self.y_predict_binned = self.y_predict_binned / 8

    def save_model(self,filepath):
        joblib.dump(self.model,filepath + ".pkl")

    def optimise(self):
        pass

    def synth_model(self):
        pass
class SklearnClassifierModel(GBDTClassifierModel):
    def __init__(self):
        from sklearn.ensemble import GradientBoostingClassifier 
        super().__init__()

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

    def synth_model(self,sim : bool = True,hdl : bool = True,hls : bool = True,test_events : int = 1000):
        import shutil

        if sim:
            if Path("simdir").is_dir():
              shutil.rmtree("simdir")

            from scipy.special import expit
            from conifer import conifer
            simcfg = conifer.backends.vhdl.auto_config()
            simcfg['Precision'] = 'ap_fixed<16,8>'
            # Set the output directory to something unique
            simcfg['OutputDir'] = "simdir/"
            simcfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
            simcfg["ClockPeriod"] = 2.78

            simmodel = conifer.model(self.model, conifer.converters.sklearn, conifer.backends.vhdl, simcfg)
            simmodel.compile()

            # Run HLS C Simulation and get the output
            if test_events == 0:
              length = -1
            else:
              length = test_events
            
            temp_decision = simmodel.decision_function(self.DataSet.X_test[0:length])
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
            hdlcfg['Precision'] = 'ap_fixed<16,8>'
            # Set the output directory to something unique
            hdlcfg['OutputDir'] = "hdldir/"
            hdlcfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
            hdlcfg["ClockPeriod"] = 2.78
            hdlmodel = conifer.model(self.model, conifer.converters.sklearn, conifer.backends.vhdl, hdlcfg)
            hdlmodel.write()
            hdlmodel.build()
        if hls:
            if Path("hlsdir").is_dir():
                shutil.rmtree("hlsdir")

            hlscfg = conifer.backends.vivadohls.auto_config()
            hlscfg['Precision'] = 'ap_fixed<16,8>'
            # Set the output directory to something unique
            hlscfg['OutputDir'] = "hlsdir/"
            hlscfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
            hlscfg["ClockPeriod"] = 2.78

            # Create and compile the model
            hlsmodel = conifer.model(self.model, conifer.converters.sklearn, conifer.backends.vivadohls, hlscfg)
            hlsmodel.write()
            hlsmodel.build()
class XGBoostClassifierModel(GBDTClassifierModel):
    def __init__(self):
        super().__init__()

        self.min_child_weight =  {"min":0,"max":10, "value":1.3690275705621135}
        self.alpha            =  {"min":0,"max":1,  "value":0.9307933560230425}
        self.early_stopping   =  {"min":1,"max":20, "value":5}
        self.learning_rate    =  {"min":0,"max":1,   "value":0.3245287291246959}
        self.n_estimators     =  {"min":0,"max":1000,   "value":100}
        self.subsample        =  {"min":0,"max":0.99,"value":0.2459092973919883	}
        self.max_depth        =  {"min":1,"max":10  ,"value":3 }
        self.gamma            =  {"min":0,"max":0.99,"value":0.0	}
        self.rate_drop        =  {"min":0,"max":1,"value":0.788588}
        self.skip_drop        =  {"min":0,"max":1,"value":0.147907}

        self.csim_predictions = {}
        self.hdl_predictions = {}
        self.hls_predictions = {}
        self.onnx_predictions = 0
        self.cpp_predictions = {}

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

    def save_model(self,filepath):
        self.model.save_model(filepath + ".json")
        pickle.dump(self.model, open(filepath + ".pkl", "wb"))
        
    def load_model(self,filepath):
        self.model = pickle.load(open(filepath + ".pkl", "rb"))

    def plot_model(self):
        import matplotlib.pyplot as plt

        #import matplotlib.pyplot as plt
        #xgb.plot_importance(self.model)
        #importances = permutation_importance(self.model, self.DataSet.X_test, self.DataSet.y_test)
        #print(importances)

        importances = self.model.get_booster().get_score(importance_type='gain').values()

        plt.close()
        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(13,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

        #ax.hist(track_data_frame['trk_matchtp_pdgid'],histtype="step",
        #                    linewidth=LINEWIDTH,
        #                    color = 'k',
        #                    density=True)
        labels = "$|\\tan(\\lambda)|$","|$z_0$|","$\\chi^2_{bend}$","# Stubs","# Missing \n Internal Stubs","$\\chi^2_{r\\phi}$","$\\chi^2_{rz}$"
        plt.bar(labels,[i/sum(importances) for i in importances], linewidth=4,edgecolor='r',color='white',width=0.5)

        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
        #for tick in ax.xaxis.get_major_ticks():
        #    tick.label1.set_horizontalalignment('center')

        dx = 1/72; dy = 0/72. 
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
        #ax.xaxis.get_major_ticks()[0].label1.set_visible(False) ## set last x tick label invisible


        plt.xticks(rotation='vertical')
        ax.grid(True)
        ax.set_xlabel("Feature",ha="right",x=1)
        ax.set_ylabel("Relative Feature Importance",ha="right",y=1)
        #ax.set_yscale("log")
        ax.xaxis.labelpad = -75

        plt.tight_layout()
        plt.savefig("feat_importance.png")

    def synth_model(self,sim : bool = True,hdl : bool = True,hls : bool = True, cpp : bool = True,onnx : bool = True,
                    test_events : int = 100000, precisions : list = None, plot : bool = True):
        import shutil
    
        precisions = ['ap_fixed<12,6>','ap_fixed<11,6>','ap_fixed<11,5>','ap_fixed<10,6>','ap_fixed<10,5>','ap_fixed<10,4>']
        xgb_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.y_predict_proba[0:test_events])
        fpr_xgb,tpr_xgb,fpr_xgb_err,tpr_xgb_err,_,__,auc_xgb,auc_xgb_err = util.CalculateROC(self.DataSet.y_test[0:test_events],self.y_predict_proba[0:test_events])
        print("xgb ROC: ",xgb_roc_auc)


        if Path("FW").is_dir():
            shutil.rmtree("FW")
        os.system("mkdir FW")

        if hdl or hls:
            with open('FW/gamma_fixed_scan.txt', 'w') as f:
                    print('Type,','Precision,','Gamma,','xgb_roc_auc,','synth_roc_auc,','LUT %,','FF %,','Latency',file=f)

        if onnx:
            if Path("FW/onnxdir").is_dir():
                shutil.rmtree("FW/onnxdir")
            self.onnx_predictions = ONNX_convert_model(self,'xgboost','FW/onnxdir',test_events=test_events)
            onnx_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.onnx_predictions)
            fpr_onnx,tpr_onnx,fpr_onnx_err,tpr_onnx_err,_,__,auc_onnx,auc_onnx_err = util.CalculateROC(self.DataSet.y_test[0:test_events],self.onnx_predictions)
            print("onnx ROC: ",onnx_roc_auc, " Ratio: ",onnx_roc_auc/xgb_roc_auc)

            ax.plot(fpr_onnx, tpr_onnx,label="ONNX AUC: %.3f $\\pm$ %.3f"%(auc_onnx,auc_onnx_err),linewidth=2,color=colours[4])
            ax.fill_between(fpr_onnx,  tpr_onnx, tpr_onnx - tpr_onnx_err,alpha=0.5,color=colours[4])
            ax.fill_between(fpr_onnx,  tpr_onnx, tpr_onnx + tpr_onnx_err,alpha=0.5,color=colours[4])

        for precision in precisions:
            start_time = time.time()

            fig, ax = plt.subplots(1,1, figsize=(10,10)) 
            hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

            print("===========================")
            print(" Precision: ",precision)
            directory_name=precision.replace('ap_fixed<', '')
            directory_name=directory_name.replace('>', '')
            directory_name=directory_name.replace(',', '')

            if sim:
                if Path("FW/sim"+directory_name).is_dir():
                    shutil.rmtree("FW/sim"+directory_name)
                simcfg = conifer.backends.xilinxhls.auto_config()
                self.csim_predictions[precision] = SynthBDTmodel(self,'xgboost',simcfg,"FW/sim"+directory_name,precision,test_events=test_events)[0]
                sim_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.csim_predictions[precision])
                print("sim ROC: ",sim_roc_auc, " Ratio: ",sim_roc_auc/xgb_roc_auc)

                fpr_sim,tpr_sim,fpr_sim_err,tpr_sim_err,_,__,auc_sim,auc_sim_err = util.CalculateROC(self.DataSet.y_test[0:test_events],self.csim_predictions[precision])
                ax.plot(fpr_sim, tpr_sim,label="C-Sim " + precision + " AUC: %.3f $\\pm$ %.3f"%(auc_sim,auc_sim_err),linewidth=2,color=colours[5])
                
                ax.fill_between(fpr_sim, tpr_sim, tpr_sim + tpr_sim_err,alpha=0.5,color=colours[5])
                ax.fill_between(fpr_sim, tpr_sim, tpr_sim - tpr_sim_err,alpha=0.5,color=colours[5])

            if hdl:
                if Path("FW/hdl"+directory_name).is_dir():
                    shutil.rmtree("FW/hdl"+directory_name)
                hdlcfg = conifer.backends.vhdl.auto_config()
                self.hdl_predictions[precision],hdlreport = SynthBDTmodel(self,'xgboost',hdlcfg,'FW/hdl'+directory_name,precision,build=True,test_events=test_events)
                hdl_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.hdl_predictions[precision])
                print("hdl ROC: ",hdl_roc_auc, " Ratio: ",hdl_roc_auc/xgb_roc_auc)
                with open('FW/gamma_fixed_scan.txt', 'a') as f:
                    print('hdl: ',precision,self.model.gamma,xgb_roc_auc,hdl_roc_auc,hdlreport["lut"]/788160,hdlreport["ff"]/1576320,hdlreport['latency'],file=f)

                fpr_hdl,tpr_hdl,fpr_hdl_err,tpr_hdl_err,_,__,auc_hdl,auc_hdl_err = util.CalculateROC(self.DataSet.y_test[0:test_events],self.hdl_predictions[precision])
                ax.plot(fpr_hdl, tpr_hdl,label="HDL " + precision + " AUC: %.3f $\\pm$ %.3f"%(auc_hdl,auc_hdl_err),linewidth=2,color=colours[1])
                ax.fill_between(fpr_hdl, tpr_hdl, tpr_hdl + tpr_hdl_err,alpha=0.5,color=colours[1])
                ax.fill_between(fpr_hdl, tpr_hdl, tpr_hdl - tpr_hdl_err,alpha=0.5,color=colours[1])

            if hls:
                if Path("FW/hls"+directory_name).is_dir():
                    shutil.rmtree("FW/hls"+directory_name)
                hlscfg = conifer.backends.xilinxhls.auto_config()
                self.hls_predictions[precision],hlsreport = SynthBDTmodel(self,'xgboost',hlscfg,'FW/hls'+directory_name,precision,build=True,test_events=test_events)
                xgb_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.y_predict_proba[0:test_events])
                hls_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.hls_predictions[precision])
                print("hls ROC: ",hls_roc_auc, " Ratio: ",hls_roc_auc/xgb_roc_auc)
                with open('FW/gamma_fixed_scan.txt', 'a') as f:
                    print('hls: ',precision,self.model.gamma,xgb_roc_auc,hls_roc_auc,hlsreport["lut"]/788160,hlsreport["ff"]/1576320,hlsreport['latency'],file=f)

                fpr_hls,tpr_hls,fpr_hls_err,tpr_hls_err,_,__,auc_hls,auc_hls_err = util.CalculateROC(self.DataSet.y_test[0:test_events],self.hls_predictions[precision])
                ax.plot(fpr_hls, tpr_hls,label="HLS " + precision + " AUC: %.3f $\\pm$ %.3f"%(auc_hls,auc_hls_err),linewidth=2,color=colours[2])
                ax.fill_between(fpr_hls, tpr_hls, tpr_hls + tpr_hls_err,alpha=0.5,color=colours[2])
                ax.fill_between(fpr_hls, tpr_hls, tpr_hls - tpr_hls_err,alpha=0.5,color=colours[2])

            if cpp:
                if Path("FW/cpp"+directory_name).is_dir():
                    shutil.rmtree("FW/cpp"+directory_name)
                cppcfg = conifer.backends.cpp.auto_config()
                self.cpp_predictions[precision],hlsreport = SynthBDTmodel(self,'xgboost',cppcfg,'FW/cpp'+directory_name,precision,build=False,test_events=test_events)
                cpp_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.cpp_predictions[precision])
                print("cpp ROC: ",cpp_roc_auc, " Ratio: ",cpp_roc_auc/xgb_roc_auc)

                fpr_cpp,tpr_cpp,fpr_cpp_err,tpr_cpp_err,_,__,auc_cpp,auc_cpp_err = util.CalculateROC(self.DataSet.y_test[0:test_events],self.cpp_predictions[precision])
                ax.plot(fpr_cpp, tpr_cpp,label="cpp " + precision + " AUC: %.3f $\\pm$ %.3f"%(auc_cpp,auc_cpp_err),linewidth=2,color=colours[3])
                ax.fill_between(fpr_cpp, tpr_cpp, tpr_cpp + tpr_cpp_err,alpha=0.5,color=colours[3])
                ax.fill_between(fpr_cpp, tpr_cpp, tpr_cpp - tpr_cpp_err,alpha=0.5,color=colours[3])

            end_time = time.time()
            print("Synthesis Complete in: ",end_time-start_time, " seconds")

            ax.plot(fpr_xgb, tpr_xgb,label="xgb AUC: %.3f $\\pm$ %.3f"%(auc_xgb,auc_xgb_err),linewidth=2,color=colours[0])
            ax.fill_between(fpr_xgb,  tpr_xgb , tpr_xgb - tpr_xgb_err,alpha=0.5,color=colours[0])
            ax.fill_between(fpr_xgb,  tpr_xgb,  tpr_xgb + tpr_xgb_err,alpha=0.5,color=colours[0])

            ax.set_xlim([0.0,1.0])
            ax.set_ylim([0.0,1.0])
            ax.set_xlabel("False Positive Rate",ha="right",x=1)
            ax.set_ylabel("Identification Efficiency",ha="right",y=1)
            ax.legend()
            ax.grid()
            plt.savefig("FW/ROC"+directory_name+".png",dpi=600)
            plt.clf()

        if plot:
            fig, ax = plt.subplots(1,1, figsize=(10,10)) 
            hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
            ax.plot(fpr_xgb, tpr_xgb,label="xgb AUC: %.3f $\\pm$ %.3f"%(auc_xgb,auc_xgb_err),linewidth=2,color=colours[0])
            ax.fill_between(fpr_xgb,  tpr_xgb , tpr_xgb - tpr_xgb_err,alpha=0.5,color=colours[0])
            ax.fill_between(fpr_xgb,  tpr_xgb,  tpr_xgb + tpr_xgb_err,alpha=0.5,color=colours[0])

            if sim:
                for i,precision in enumerate(precisions):                         
                    fpr_sim,tpr_sim,fpr_sim_err,tpr_sim_err,_,__,auc_sim,auc_sim_err = util.CalculateROC(self.DataSet.y_test[0:test_events],self.csim_predictions[precision])
                    ax.plot(fpr_sim, tpr_sim,label="sim " + precision + " AUC: %.3f $\\pm$ %.3f"%(auc_sim,auc_sim_err),linewidth=2,color=colours[i+1])
                    ax.fill_between(fpr_sim, tpr_sim, tpr_sim + tpr_sim_err,alpha=0.5,color=colours[i+1])
                    ax.fill_between(fpr_sim, tpr_sim, tpr_sim - tpr_sim_err,alpha=0.5,color=colours[i+1])

            ax.set_xlim([0.0,1.0])
            ax.set_ylim([0.0,1.0])
            ax.set_xlabel("False Positive Rate",ha="right",x=1)
            ax.set_ylabel("Identification Efficiency",ha="right",y=1)
            ax.legend()
            ax.grid()
            plt.savefig("FW/ROC_scan.png",dpi=600)
            plt.clf()


class FullXGBoostClassifierModel(XGBoostClassifierModel):
    def __init__(self):
        super().__init__()

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
        self.dtrain = xgb.DMatrix(self.DataSet.X_train.to_numpy(),label=np.ravel(self.DataSet.y_train))
        self.dtest = xgb.DMatrix(self.DataSet.X_test.to_numpy(),label=np.ravel(self.DataSet.y_test))

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

    def plot_model(self):
        xgb.plot_importance(self.model)

    def synth_model(self,sim : bool = False,hdl : bool = True,hls : bool = False,test_events : int = 1000,intwidth : int = 32,fracwidth : int = 16, plot : bool = False, reportname : str = 'gamma_fixed_scan.txt'):
        import shutil
        import conifer
        from scipy.special import expit

        if sim:
            if Path("simdir").is_dir():
              shutil.rmtree("simdir")


            simcfg = conifer.backends.xilinxhls.auto_config()
            simcfg['Precision'] = 'ap_fixed<'+str(intwidth)+','+str(fracwidth)+'>'
            # Set the output directory to something unique
            simcfg['OutputDir'] = "simdir/"
            simcfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
            simcfg["ClockPeriod"] = 2.78

            self.simmodel = conifer.converters.convert_from_xgboost(self.model, simcfg)
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
                
                simauc = metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.synth_y_predict_proba)
                xgbsim = metrics.roc_auc_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.y_predict_proba[0:len(self.synth_y_predict_proba)])
                

                print("AUC ROC sim: {}".format(simauc))
                print("AUC ROC xgb: {}".format(xgbsim))
                print("Accuracy sim: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.synth_y_predict)))
                print("Accuracy sklearn: {}".format(metrics.accuracy_score(self.DataSet.y_test[0:len(self.synth_y_predict_proba)],  self.y_predict[0:len(self.synth_y_predict_proba)])))

                if plot:
                    fig, ax = plt.subplots(1,1, figsize=(18,18)) 
                    ax.tick_params(axis='x', labelsize=16)
                    ax.tick_params(axis='y', labelsize=16)
                    ax.tick_params(axis='x', labelsize=16)
                    ax.tick_params(axis='y', labelsize=16)
                    fprsim, tprsim, thresholdssim = metrics.roc_curve(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.synth_y_predict_proba, pos_label=1)
                    fprxgb, tprxgb, thresholdsxgb = metrics.roc_curve(self.DataSet.y_test[0:len(self.synth_y_predict_proba)], self.y_predict_proba[0:len(self.synth_y_predict_proba)], pos_label=1)
                    ax.set_title("ROC Curve",loc='left',fontsize=20)
                    ax.plot(fprsim,tprsim,label="Conifer Sim <"+str(intwidth)+","+str(fracwidth)+">" + "AUC: %.3f"%simauc)
                    ax.plot(fprxgb,tprxgb,label="XGBoost AUC: %.3f"%simauc)
                    ax.set_xlim([0,0.3])
                    ax.set_ylim([0.8,1.0])
                    ax.set_xlabel("Fake Positive Rate",ha="right",x=1)
                    ax.set_ylabel("Identification Efficiency",ha="right",y=1)
                    ax.legend()
                    ax.grid()
                    plt.savefig("SimAUC.png",dpi=600)


        if hdl:

            precisions = ['ap_fixed<12,6>']

            for precision in precisions:
                if Path("hdldir").is_dir():
                    shutil.rmtree("hdldir")
                hdlcfg = conifer.backends.vhdl.auto_config()
                hdlcfg['Precision'] = precision
                # Set the output directory to something unique
                hdlcfg['OutputDir'] = "hdldir/"
                hdlcfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
                hdlcfg["ClockPeriod"] = 2.78
                hdlmodel = conifer.converters.convert_from_xgboost(self.model,hdlcfg)
                hdlmodel.compile()

                print("Compiled")
                length = 1000
                y_hls = expit(hdlmodel.decision_function(self.DataSet.X_test.to_numpy()[0:length]))
                y_xgb = (self.model.predict_proba(self.DataSet.X_test.to_numpy()[0:length])[:,1])

                xgb_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:length],y_xgb)
                hls_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:length],y_hls)

                #print("AUC ROC xgb: {}".format(xgb_roc_auc))
                #print("AUC ROC hls: {}".format(hls_roc_auc))

                hdlmodel.write()
                hdlmodel.build()
                report = hdlmodel.read_report()

                                
                with open(reportname, 'a') as f:
                    print('\n',precision,self.model.gamma,xgb_roc_auc,hls_roc_auc,0.9801332346410067,report["lut"]/788160,report["ff"]/1576320,report['latency'],file=f)

        if hls:
            if Path("hlsdir").is_dir():
                shutil.rmtree("hlsdir")

            hlscfg = conifer.backends.vivadohls.auto_config()
            hlscfg['Precision'] = 'ap_fixed<32,16>'
            # Set the output directory to something unique
            hlscfg['OutputDir'] = "hlsdir/"
            hlscfg["XilinxPart"] = 'xcvu13p-flga2577-2-e'
            hlscfg["ClockPeriod"] = 2.78

            # Create and compile the model
            #hlsmodel = conifer.model(self.model.get_booster(), conifer.converters.xgboost, conifer.backends.vivadohls, hlscfg)
            hlsmodel = conifer.converters.convert_from_xgboost(self.model,hlscfg)
            hlsmodel.compile()


            # Run HLS C Simulation and get the output
            # xgboost 'predict' returns a probability like sklearn 'predict_proba'
            # so we need to compute the probability from the decision_function returned
            # by the HLS C Simulation
            y_hls = hlsmodel.decision_function(self.DataSet.X_test.to_numpy())
            #y_xgb = self.model.predict_proba(self.DataSet.X_test.to_numpy())

            xgb_roc_auc = metrics.roc_auc_score(self.DataSet.y_test,self.y_predict_proba)
            hls_roc_auc = metrics.roc_auc_score(self.DataSet.y_test,y_hls)

            print("AUC ROC xgb: {}".format(xgb_roc_auc))
            print("AUC ROC hls: {}".format(hls_roc_auc))

            hlsmodel.write()
            hlsmodel.build()
            hlsmodel.read_report()
class TFDFClassifierModel(GBDTClassifierModel):
    def __init__(self):
        super().__init__()

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
        self.train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.DataSet.X_train,"label")
        self.test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.DataSet.X_test,"label")

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

    def plot_model(self):
        tfdf.model_plotter.plot_model_in_colab(self.model, tree_idx=0, max_depth=3)

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import mplhep as hep
        #hep.set_style("CMSTex")

        hep.cms.label()
        hep.cms.text("Simulation")
        plt.style.use(hep.style.CMS)

        SMALL_SIZE = 20
        MEDIUM_SIZE = 25
        BIGGER_SIZE = 30

        LEGEND_WIDTH = 20
        LINEWIDTH = 3
        MARKERSIZE = 20

        colormap = "jet"

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('axes', linewidth=LINEWIDTH)              # thickness of axes
        plt.rc('xtick', labelsize=SMALL_SIZE+2)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE-2)            # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        matplotlib.rcParams['xtick.major.size'] = 10
        matplotlib.rcParams['xtick.major.width'] = 3
        matplotlib.rcParams['xtick.minor.size'] = 10
        matplotlib.rcParams['xtick.minor.width'] = 3

        matplotlib.rcParams['ytick.major.size'] = 20
        matplotlib.rcParams['ytick.major.width'] = 3
        matplotlib.rcParams['ytick.minor.size'] = 10
        matplotlib.rcParams['ytick.minor.width'] = 2
        importances = self.model.get_booster().get_score(importance_type='gain').values()

        plt.close()
        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(13,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)

        labels = "$|\\tan(\\lambda)|$","|$z_0$|","$\\chi^2_{bend}$","# Stubs","# Missing \n Internal Stubs","$\\chi^2_{r\\phi}$","$\\chi^2_{rz}$"
        plt.bar(labels,[i/sum(importances) for i in importances], linewidth=4,edgecolor='r',color='white',width=0.5)

        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
        #for tick in ax.xaxis.get_major_ticks():
        #    tick.label1.set_horizontalalignment('center')

        dx = 1/72; dy = 0/72. 
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
        #ax.xaxis.get_major_ticks()[0].label1.set_visible(False) ## set last x tick label invisible


        plt.xticks(rotation='vertical')
        ax.grid(True)
        ax.set_xlabel("Feature",ha="right",x=1)
        ax.set_ylabel("Relative Feature Importance",ha="right",y=1)
        #ax.set_yscale("log")
        ax.xaxis.labelpad = -75

        plt.tight_layout()
        plt.savefig("feat_importance.png")

    def synth_model(self,sim : bool = True,hdl : bool = True,hls : bool = False, cpp : bool = False, test_events : int = 1000,intwidth : int = 32,fracwidth : int = 16, plot : bool = False,dir=None):
        import shutil
        import conifer
        from scipy.special import expit

        self.inspector = tfdf.inspector.make_inspector(os.path.join("Models/NoDegredation", "assets"))

        precisions = ['ap_fixed<12,6>']
        #xgb_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.y_predict_proba[0:test_events])
        #print("xgb ROC: ",xgb_roc_auc)

        #if hdl or hls:
        #    with open('gamma_fixed_scan.txt', 'w') as f:
        #            print('Type','Precision','Gamma','xgb_roc_auc','synth_roc_auc','LUT %','FF %','Latency',file=f)
        for precision in precisions:

            if sim:
                if Path("simdir").is_dir():
                    shutil.rmtree("simdir")
                simcfg = conifer.backends.xilinxhls.auto_config()
                print(simcfg)
                self.csim_predictions = SynthBDTmodel(self,'tfdf',simcfg,'simdir',precision)[0]
                sim_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.csim_predictions)
                print("sim ROC: ",sim_roc_auc)

            if hdl:
                if Path("hdldir").is_dir():
                    shutil.rmtree("hdldir")
                hdlcfg = conifer.backends.vhdl.auto_config()
                self.hdl_predictions,hdlreport = SynthBDTmodel(self,'tfdf',hdlcfg,'hdldir',precision,build=True)
                hdl_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.hdl_predictions)
                print("hdl ROC: ",hdl_roc_auc)
                with open('gamma_fixed_scan.txt', 'a') as f:
                    print('hdl: ',precision,self.model.gamma,xgb_roc_auc,hdl_roc_auc,hdlreport["lut"]/788160,hdlreport["ff"]/1576320,hdlreport['latency'],file=f)

            if hls:
                if Path("hlsdir").is_dir():
                    shutil.rmtree("hlsdir")
                hlscfg = conifer.backends.xilinxhls.auto_config()
                self.hls_predictions,hlsreport = SynthBDTmodel(self,'tfdf',hlscfg,'hlsdir',precision,build=True)
                xgb_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.y_predict_proba[0:test_events])
                hls_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.hls_predictions)
                print("hls ROC: ",hls_roc_auc)
                with open('gamma_fixed_scan.txt', 'a') as f:
                    print('hls: ',precision,self.model.gamma,xgb_roc_auc,hls_roc_auc,hlsreport["lut"]/788160,hlsreport["ff"]/1576320,hlsreport['latency'],file=f)

            if cpp:
                if Path("cppdir").is_dir():
                    shutil.rmtree("cppdir")
                cppcfg = conifer.backends.cpp.auto_config()
                self.cpp_predictions,hlsreport = SynthBDTmodel(self,'tfdf',cppcfg,'cppdir',precision,build=False)
                cpp_roc_auc = metrics.roc_auc_score(self.DataSet.y_test[0:test_events],self.cpp_predictions)
                print("cpp ROC: ",cpp_roc_auc)