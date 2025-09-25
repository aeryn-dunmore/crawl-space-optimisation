#!/usr/bin/env python3

#!/usr/bin/env python3
#import create_test_sets
from keras.src import models
import keras
import sys
import os
from datetime import datetime
import traceback
import cso_clean as run_opt

speech_type = "mel_zcr"

params = { "nums":2, "train":True, "test":True, "pickle":False, "run_all_train": False,
	"overall_val":0, "overall_act":0, "labels":["valence","arousal"], "full_dir":"","multioutput":True, "mean":True, "test_num":"", "method":"mel", "img":"feature_extract",
	"results_dir":"", "flatten":False, "img_processing":"dft", "signal_type":"dft", 
	"get_from_list":True,
	"dataset_dir":"", "dataset":"IEMOCAP", "cnn":False, "scale":1, "use_dominance":False, 
	"combine_bios":False, "individual_scores": True, "combine_here":True, "model":"",
}
def drange(x, y, jump):
	while x < y:
		yield float(x)
		x += float(jump)
		
def write_all(files,text):
	for f in files:
		f.write(text)
		f.flush()


params["dataset"] = #"IEMOCAP" or "K-EmoCon"
current_wd = os.getcwd()

results_dir = current_wd+"/"+params["dataset"]+" Optimisation Trials/Continuous/Single Signal/"
params["results_dir"] = results_dir
params["output_dir"] = current_wd+"/Continuous/"
params["full_dir"] = results_dir
params["iterations"] = 100
speech_dir = current_wd+"/"
dataset_dir = current_wd+"/Continuous/"
params["dataset_dir"] = current_wd+"/Continuous/"

best_model = 10
params["scale"] = 1
params["iterative_options"] = {}
params["iterative_options"]["token_type"] = [1,2,3,4]
params["iterative_options"]["model_type"] = ["MLP","LSTM","RNN","BiRNN","XGBoost","ERT","CNN"] #"DT","RF",
params["iterative_options"]["optimiser"] = ["Adam","SGD","RMSprop","Adadelta","Adagrad","Adamax","Nadam","Ftrl"]
params["iterative_options"]["features"] = ["mel_zcr","NA","mfcc_deltas","mel"]
params["iterative_options"]["images"] = ["raw","spectrogram","mel-spectrogram","cochleagram"]#,"raw"
params["iterative_params"] = ["token_type","features","images","model_type","optimiser"]

params["mode"] = "speech"
sys.stdout.flush()
params["interval"] = 5
#features = speech_features
params["points_list"] = {
	"epochs": list(range(5,25)),
	"neurons":list(range(10,50)),
  	"lr":list(drange(0.0001,0.2,0.00001)),
   	"gamma":list(drange(0.0,0.01,0.00001)),
	"max_depth":list(range(3,10)),
	"min_cweight":list(range(1,5)),
	"subsample":list(drange(0.5,0.8,0.001)),
	"n_estimators":list(range(100,800)),
	"colsamples":list(drange(0.3,0.9,0.001)),
	"hidden":list(range(1,5)),
}
params["bounds_list"] = {
	"lr":[0.00001,0.2],
 	"neurons":[10,200],
  	"epochs":[5,150],
   	"gamma":[0,0.1],
	"max_depth":[1,15],
	"min_cweight":[1,10],
	"subsample":[0.4,0.95],
	"n_estimators":[50,1000],
	"colsamples":[0.1,0.95],
	"hidden":[1,10],
}
params["int_val"] = {
	"lr":False,
	"neurons":True,
	"epochs":True,
	"gamma":False,
	"max_depth":True,
	"min_cweight":False,
	"subsample":False,
	"n_estimators":True,
	"colsamples":False,
	"hidden":True,
}

params["all_trials"] = 0
run_to_solution = False
params["convergence"] = 0.05
maximum_runs = 100
params["rate_of_change"] = 10


params["metrics"] = ["f1","mae","mse","mbe","dir"]
converge = "Y"
params["speech_labels"] = ["label"]
params["model_name"] = "continuous_individual"
tries = 5
params["categorical"] = False
params["to_optimise"] = ["lr","neurons","epochs","hidden","gamma","max_depth","min_cweight","subsample","n_estimators","colsamples"]

try:
	run_opt.main(params)
except Exception as e:
	print("Traceback:")
	print(traceback.format_exc())
