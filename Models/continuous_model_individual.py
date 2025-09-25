# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#!/usr/bin/env python3

#!/usr/bin/env python3

# This runs on the speech from KEmoCon dataset.
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from sklearn import tree
import pickle
import signal
from sklearn.utils import shuffle

import os
import matplotlib.pyplot as plt
import sys
import decimal
from keras.layers import Bidirectional, TimeDistributed
from keras import layers
from tensorflow import keras
from threading import Timer


global results
global test_results
global model_names
global params
params = { "nums":2, "train":True, "test":True, "pickle":False,
	"overall_val":0, "overall_act":0, "labels":["valence","arousal"], "full_dir":"/Volumes/PhD Data/Emotion and Speech/Speech Tests and Code/Continuous/",
	"multioutput":True, "mean":False, "method":"mel", "img":"feature_extract",
	"results_dir":"/Volumes/PhD Data/Emotion and Speech/Speech Tests and Code/Continuous/", "flatten":False, "img_processing":"dft", "signal_type":"dft", 
	"get_from_list":True,"type":1,
	"dataset_dir":"/Volumes/PhD Data/Emotion and Speech/KEmo-Con/", "dataset":"IEMOCAP", 
	"use_dominance":False, "combine_bios":False, "individual_scores": True, "combine_here":False, "speech_only":True, "optimiser":"Adam",
}		

global weights
weights = [0.5,0.5,1,1,1,1,1,1,1,1,1,1]
global models
models = {}
global predictions
predictions = []

class RepeatedTimer:
	def __init__(self, interval, function, *args, **kwargs):
		self._timer = None
		self.interval = interval
		self.function = function
		self.args = args
		self.kwargs = kwargs
		self.is_running = False
		self.daemon = True
		self.start()

	def _run(self):
		self.is_running = False
		self.start()
		self.function(*self.args, **self.kwargs)

	def start(self):
		if not self.is_running:
			self._timer = Timer(self.interval, self._run)
			self._timer.start()
			self.is_running = True

	def stop(self):
		if self._timer:
			self._timer.cancel()
			self.is_running = False

## Flush the buffer every  set seconds.
def refresh():
	sys.stdout.flush()

def check_memory():
	print(f"CPU Percent: {psutil.cpu_percent()}")
	# you can convert that object to a dictionary 
	print(dict(psutil.virtual_memory()._asdict()))
	# you can have the percentage of used RAM
	print(f"Virtual Memory Percentage: {psutil.virtual_memory().percent}")
	# you can calculate percentage of available memory
	print(f"Percentage of Available Memory Used: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total}")
	sys.stdout.flush()


def drange(x, y, jump):
	while x < y:
		yield float(x)
		x += float(jump)
		
def normalise_column_max_min(col):
	normalised = (col-col.min())/(col.max()-col.min())
	return normalised

def get_iqr(predictions,actual):
	global model_name,dir
	error_range = np.abs(predictions,actual)
	va_error_range = np.abs(predictions-actual)
	valence_error_range = va_error_range[:,0]
	arousal_error_range = va_error_range[:,1]
	bins = list(drange(0, 1, decimal.Decimal('0.05')))
	plt.tight_layout()
	plt.subplot(325)
	plt.xlabel("Valence Error Values")
	plt.ylabel("Frequency")
#	weights_actual = np.zeros_like(np.reshape(ytest,(-1)) + 100. / len(np.reshape(ytest,(-1))))
	plt.hist(valence_error_range,bins = bins,color=["yellow"],label="actual",density=True)
	plt.tight_layout()
	plt.subplot(326)
	plt.xlabel("Arousal Error Values")
	plt.ylabel("Frequency")
#	weights_actual = np.zeros_like(np.reshape(ytest,(-1)) + 100. / len(np.reshape(ytest,(-1))))
	plt.hist(arousal_error_range,bins = bins,color=["yellow"],label="actual",density=True)
	sorted_error = np.sort(error_range)
	q2 = np.median(sorted_error)
	middle = len(sorted_error)/2
	if middle%1 != 0:
		q1_end = int(middle)
		q3_start = q1_end+1
	else:
		q1_end = int(middle-1)
		q3_start = int(middle+1)
	q1 = sorted_error[0:q1_end]
	q3 = sorted_error[q3_start:]
	q1_med = np.median(q1)
	q3_med = np.median(q3)
	iqr = q3_med-q1_med
	v_q2 = np.median(valence_error_range)
	a_q2 = np.median(arousal_error_range)
	sorted_v = np.sort(valence_error_range)
	sorted_a = np.sort(arousal_error_range)
	v_middle = len(sorted_v)/2
	if v_middle%1 != 0:
		q1_end = int(v_middle)
		q3_start = q1_end+1
	else:
		q1_end = int(v_middle-1)
		q3_start = int(v_middle+1)
	v_q1 = sorted_v[0:q1_end]
	v_q3 = sorted_v[q3_start:]
	v_q1_med = np.median(v_q1)
	v_q3_med = np.median(v_q3)
	v_iqr = v_q3_med-v_q1_med
	a_middle = len(sorted_a)/2
	if a_middle%1 != 0:
		q1_end = int(a_middle)
		q3_start = q1_end+1
	else:
		q1_end = int(a_middle-1)
		q3_start = int(a_middle+1)
	a_q1 = sorted_a[0:q1_end]
	a_q3 = sorted_a[q3_start:]
	a_q1_med = np.median(a_q1)
	a_q3_med = np.median(a_q3)
	a_iqr = a_q3_med-a_q1_med
	print(f"Valence IQR: {v_iqr} ({v_q2}), Arousal IQR: {a_iqr} ({a_q2})")
	print(f"Median for all errors: {q2}\nIQR for all errors: {iqr}")
	return iqr

def create_hist(predictions_this,ytest,model_name,dir):
	print(ytest[0:5])
	print(predictions_this[0:5])
	return 0
	# ytest = np.asarray(ytest).reshape(-1)
	# predictions_this = np.asarray(predictions_this).reshape(-1)
	valence_pred = predictions_this[:,0]
	arousal_pred = predictions_this[:,1]
	valence_actual = ytest[:,0]
	arousal_actual = ytest[:,1]
	bins = list(drange(0, 1, decimal.Decimal('0.1')))
	plt.tight_layout()
	plt.subplot(321)
	plt.title("Predictions vs Actual Values with Error Rates")
	plt.ylabel("Frequency")
	plt.xlabel("Valence Predictions")
	plt.tight_layout()
#	weights_pred = np.zeros_like(np.reshape(predictions_this,(-1)) + 100. / len(np.reshape(predictions_this,(-1))))
	plt.hist(valence_pred,bins = bins,color="blue",label="predictions",density=True)
	plt.tight_layout()
	plt.subplot(322)
	plt.ylabel("Frequency")
	plt.xlabel("Arousal Predictions")
	plt.tight_layout()
#	weights_pred = np.zeros_like(np.reshape(predictions_this,(-1)) + 100. / len(np.reshape(predictions_this,(-1))))
	plt.hist(arousal_pred,bins = bins,color="blue",label="predictions",density=True)
	
	plt.tight_layout()
	plt.subplot(323)
	plt.xlabel("Actual Valence Values")
	plt.ylabel("Frequency")
#	weights_actual = np.zeros_like(np.reshape(ytest,(-1)) + 100. / len(np.reshape(ytest,(-1))))
	plt.hist(valence_actual,bins = bins,color="red",label="actual",density=True)
	plt.tight_layout()
	plt.subplot(324)
	plt.ylabel("Frequency")
	plt.xlabel("Actual Arousal Values")
	plt.tight_layout()
#	weights_pred = np.zeros_like(np.reshape(predictions_this,(-1)) + 100. / len(np.reshape(predictions_this,(-1))))
	plt.hist(arousal_actual,bins = bins,color="red",label="predictions",density=True)
	
	plt.savefig(dir+model_name+"_pred_vs_actual_hist.png",format="png")
	
def get_metrics(predictions_this,ytest):
	global mse,mbe
	mae = mean_absolute_error(ytest,predictions_this)
	mse = mean_squared_error(ytest,predictions_this)
	rmse = np.sqrt(mse)
#	mape = mean_absolute_percentage_error(ytest,predictions_this)
	mbe = np.mean(ytest-predictions_this)
	# iqr = get_iqr(predictions_this,ytest)
	return mae,mse,rmse,mbe

def random_forest(params):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir
	xtrain = np.asarray(xtrain,dtype="float32")
	print(f"Training data shape: {np.shape(xtrain)}")
	shape = np.shape(xtrain)
	if len(shape) > 2:
		xtrain = np.reshape(xtrain,[shape[0],-1])
	ytrain = np.asarray(ytrain)
	xtest = np.asarray(xtest,dtype="float32")
	shape = np.shape(xtest)
	if len(shape) > 2:
		xtest = np.reshape(xtest,[shape[0],-1])
	model = RandomForestRegressor(max_depth=params["max_depth"],n_estimators=params["n_estimators"],n_jobs=-1,verbose=2)
	model.fit(xtrain,ytrain)
	f = open(dir+model_name+"_rf.pickle","wb")
	pickle.dump(model,f)
	f.close()
	predictions_this = model.predict(xtest)
	global mse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe

def decision_tree(params):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir
	xtrain = np.asarray(xtrain,dtype="float32")
	print(f"Training data shape: {np.shape(xtrain)}")
	shape = np.shape(xtrain)
	if len(shape) > 2:
		xtrain = np.reshape(xtrain,[shape[0],-1])
	ytrain = np.asarray(ytrain)
	xtest = np.asarray(xtest,dtype="float32")
	shape = np.shape(xtest)
	if len(shape) > 2:
		xtest = np.reshape(xtest,[shape[0],-1])
	model = tree.DecisionTreeRegressor(max_depth=params["neurons"],max_leaf_nodes=params["neurons"])
	# grid_pred,model = search(model,param_grid,xtrain,ytrain,xtest,ytest)
	f = open(dir+model_name+"_dt.pickle","wb")
	pickle.dump(model,f)
	f.close()
	predictions_this = model.predict(xtest)
	global mse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe

def nearest_neighbour(params):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir
	xtrain = np.asarray(xtrain,dtype="float32")
	print(f"Training data shape: {np.shape(xtrain)}")
	shape = np.shape(xtrain)
	if len(shape) > 2:
		xtrain = np.reshape(xtrain,[shape[0],-1])
	ytrain = np.asarray(ytrain)
	xtest = np.asarray(xtest,dtype="float32")
	shape = np.shape(xtest)
	if len(shape) > 2:
		xtest = np.reshape(xtest,[shape[0],-1])
	model = tree.DecisionTreeRegressor()
	model.fit(xtrain,ytrain)
	# grid_pred,model = search(model,param_grid,xtrain,ytrain,xtest,ytest)
	f = open(dir+model_name+"_knn.pickle","wb")
	pickle.dump(model,f)
	f.close()
	predictions_this = model.predict(xtest)
	global mse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe

def BiLSTM(params):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir,optimisers
	out = np.shape(ytrain)[1]
	xtrain = np.asarray(xtrain,dtype="float32")
	ytrain = np.asarray(ytrain)
	xtest = np.asarray(xtest,dtype="float32")
	model = keras.Sequential()
	model.add(Bidirectional(layers.LSTM(params["neurons"],dropout=params["dropout"])))
#	model.add(layers.LSTM(128))
	model.add(layers.Dense(out))
	
	model.compile(optimizer=optimisers[params["optimiser"]][params["lr"]],loss=keras.losses.MeanSquaredError(),metrics=["mae"])
	model.build()
	model.fit(xtrain,ytrain,epochs=params["epochs"])
	# grid_pred,model = search(model,param_grid,xtrain,ytrain,xtest,ytest)
	print(model.summary())
	predictions_this = model.predict(xtest)
#	print(predictions_this)
	f = open(dir+model_name+"_bilstm.pickle","wb")
	pickle.dump(model,f)
	f.close()
#	print(np.shape(predictions_this),np.shape(ytest))
	global mse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe

def LSTM(params):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir,optimisers
	out = np.shape(ytrain)[1]
	xtrain = np.asarray(xtrain,dtype="float32")
	ytrain = np.asarray(ytrain)
	xtest = np.asarray(xtest,dtype="float32")
	model = keras.Sequential()
	model.add(layers.LSTM(params["neurons"],dropout=params["dropout"]))
	model.add(layers.Dense(out))
	# model.compile(loss=keras.losses.MeanSquaredError(),metrics=["mae"])
	model.compile(optimizer=optimisers[params["optimiser"]][params["lr"]],loss=keras.losses.MeanAbsoluteError(),metrics=["mae","mse"])
	model.fit(xtrain,ytrain,epochs=params["epochs"])
	# grid_pred,model = search(model,param_grid,xtrain,ytrain,xtest,ytest)
	f = open(dir+model_name+"_lstm.pickle","wb")
	pickle.dump(model,f)
	f.close()
	print(model.summary())
	predictions_this = model.predict(xtest)
#	print(predictions_this)
	
	print(np.shape(predictions_this),np.shape(ytest))
	global mae,mse,rmse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe

def BiRNN(params):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir,optimisers
	out = np.shape(ytrain)[1]
	xtrain = np.asarray(xtrain,dtype="float32")
	ytrain = np.asarray(ytrain)
	xtest = np.asarray(xtest,dtype="float32")
	model = keras.Sequential()
	
	model.add(
		layers.Bidirectional(layers.LSTM(params["neurons"],dropout=params["dropout"]))
	)
	model.add(layers.Bidirectional(layers.LSTM(32)))
	model.add(layers.Dense(out))
	model.compile(optimizer=optimisers[params["optimiser"]][params["lr"]],loss=keras.losses.MeanAbsoluteError(),metrics=["mae"])
	model.fit(xtrain,ytrain,epochs=params["epochs"])
	# grid_pred,model = search(model,param_grid,xtrain,ytrain,xtest,ytest)
	#	
	model.summary()
	f = open(dir+model_name+"_birnn.pickle","wb")
	pickle.dump(model,f)
	f.close()
	predictions_this = model.predict(xtest)
#	print(np.shape(predictions_this),np.shape(ytest))
	global mae,mse,rmse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe


def RNN_simple(params):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir,optimisers
	out = np.shape(ytrain)[1]
	xtrain = np.asarray(xtrain,dtype="float32")
	ytrain = np.asarray(ytrain)
	xtest = np.asarray(xtest,dtype="float32")
	input_shape = np.shape(xtrain)[1:]
	model = keras.Sequential()
	model.add(layers.SimpleRNN(params["neurons"],dropout=params["dropout"]))
	model.add(layers.Dense(out))
	model.compile(loss=keras.losses.MeanAbsoluteError(),metrics=["mae"],optimizer=optimisers[params["optimiser"]](params["lr"]))
	model.fit(xtrain,ytrain,batch_size=params["n_estimators"],epochs=params["epochs"])
	# grid_pred,model = search(model,param_grid,xtrain,ytrain,xtest,ytest)
	print(model.summary())
	# f = open(dir+model_name+"_rnn.pickle","wb")
	# pickle.dump(model,f)
	# f.close()
	predictions_this = model.predict(xtest)
#	print(np.shape(predictions_this),np.shape(ytest))
	global mae,mse,rmse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	create_hist(predictions_this,ytest,model_name,dir)
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe


def Time_LSTM(params):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir,optimisers
	out = np.shape(ytrain)[1]
	shapes = np.shape(xtrain)[1:]
	xtrain = np.asarray(xtrain,dtype="float32")
	print(np.shape(xtrain))
	ytrain = np.asarray(ytrain)
	xtest = np.asarray(xtest,dtype="float32")
	model = keras.Sequential()
	# model.add(layers.InputLayer(shape=shapes))
	model.add(TimeDistributed(layers.LSTM(params["neurons"],dropout=params["dropout"])))
	model.add(layers.Dropout(params["dropout"]))
	model.add(layers.LSTM(32))
	model.add(layers.Dense(out))
	model.compile(loss=keras.losses.MeanAbsoluteError(),metrics=["mae"],optimizer=optimisers[params["optimiser"]](params["lr"]))
	model.fit(xtrain,ytrain,epochs=params["epochs"])
#	model.build()
	# grid_pred,model = search(model,param_grid,xtrain,ytrain,xtest,ytest)
	model.summary()
	f = open(dir+model_name+"_time_lstm.pickle","wb")
	pickle.dump(model,f)
	f.close()
	predictions_this = model.predict(xtest)
#	print(np.shape(predictions_this),np.shape(ytest))
	global mae,mse,rmse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe


def random_forest_regressor(param_grid):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir,optimisers
	
	reg = RandomForestRegressor(criterion="absolute_error",n_jobs=-1,verbose=2)
	reg.fit(xtrain,ytrain)
	# grid_pred,model = search(reg,param_grid,xtrain,ytrain,xtest,ytest)
	# score = reg.score(xtrain,ytrain)
	predictions_this = reg.predict(xtest)
#	predictions["Valence"].append(predictions)
	models[model_name] = reg
	f = open(params["results_dir"]+model_name+".pickle","wb")
	pickle.dump(reg,f)
	f.close()
	global mae,mse,rmse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe



def linear_regressor_model(param_grid):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir,optimisers	
	reg = LinearRegression(n_jobs=-1)
	reg.fit(xtrain,ytrain)
	# grid_pred,reg = search(reg,param_grid,xtrain,ytrain,xtest,ytest)
	# r2_1 = reg.score(xtrain,ytrain)
	predictions_this = reg.predict(xtest)
	# r2_2 = reg.score(xtest,ytest)
	models[model_name] = reg
	f = open(params["results_dir"]+model_name+".pickle","wb")
	pickle.dump(reg,f)
	f.close()
	global mae,mse,rmse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe


def multioutput_regression(param_grid):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir,optimisers	
	lin = LinearRegression(n_jobs=-1)
	reg = MultiOutputRegressor(lin,n_jobs=-1)
	reg.fit(xtrain,ytrain)
	# grid_pred,reg = search(reg,param_grid,xtrain,ytrain,xtest,ytest)
	predictions_this = reg.predict(xtest)
	predictions.append(predictions_this)
	models[model_name] = reg
	f = open(dir+model_name+".pickle","wb")
	pickle.dump(reg,f)
	f.close()
	global mae,mse,rmse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe


def cnn_regression(params):
	global xtrain,ytrain,xtest,ytest,results,model_name,dir,optimisers	
	outs = 2
	if params["use_dominance"]:
		outs = 3
	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(params["neurons"],params["neurons"], activation='relu'),
		tf.keras.layers.Conv2D(8,8, activation='relu'),
		tf.keras.layers.MaxPooling2D(2),
		tf.keras.layers.Conv2D(6, 6, activation='relu'),
		tf.keras.layers.Conv2D(6, 6, activation='relu'),
		tf.keras.layers.MaxPooling2D(2),
		tf.keras.layers.Conv2D(3, 3, activation='relu'),
		tf.keras.layers.Conv2D(3, 3, activation='relu'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(units=512, activation='relu'),
		tf.keras.layers.Dense(units=256, activation='relu'),
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=outs)
	])
	xtrain = np.array(xtrain,dtype='float32')
#	xtrain = np.expand_dims(xtrain,axis=0).asfloat()
	model.compile(loss='mean_squared_error', optimizer=optimisers[params["optimiser"]](params["lr"]))
	model.fit(xtrain,ytrain,epochs=params["epochs"])
	# grid_pred,model = search(model,param_grid,xtrain,ytrain,xtest,ytest)
	predictions_this = model.predict(np.array(xtest,dtype='float32'))
	print(model.summary())
	f = open(dir+model_name+"_cnn.pickle","wb")
	pickle.dump(model,f)
	f.close()
	global mae,mse,rmse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe


def MLP_simple(params):
	global xtrain,ytrain,xtest,ytest,results,model_name
	in_shape = np.shape(xtrain)[0]
	xtrain = np.reshape(xtrain,(in_shape,-1))
	in_shape = np.shape(xtest)[0]
	xtest = np.reshape(xtest,(in_shape,-1))
#	print(in_shape)
#	xtrain = np.transpose(xtrain)
#	length = np.shape(xtrain)[2]
	model = MLPRegressor(solver='lbfgs',activation='tanh',learning_rate='adaptive',verbose=True)
	model.fit(xtrain,ytrain)
	# grid_pred,model = search(model,param_grid,xtrain,ytrain,xtest,ytest)
	predictions_this = model.predict(xtest)
	f = open(dir+model_name+"_mlp.pickle","wb")
	pickle.dump(model,f)
	f.close()
	global mae,mse,rmse,mbe
	mae,mse,rmse,mbe = get_metrics(predictions_this,ytest)
	create_hist(predictions_this,ytest,model_name,dir)
	print(f"Mean Absolute Error: {mae}")
	print(f"Mean Squared Error: {mse}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"Mean Bias Error: {mbe}")
	r2 = r2_score(ytest,predictions_this)
	return r2,mae,mse,mbe

def sum_is_not_zero(x):
	return sum(abs(x))-1

def sum_is_one(x):
	return 1-sum(x)

def weight_is_not_zero(x):
	t = 1
	for i in x:
		if i == 0:
			t+=0.1
		if i > 0:
			t-=x
	return t

def find_weights(w,x):
	total = 0
	for i in range(len(x)):
		total+=(x[i]*w[i])
	return total

def find_overall_weights(w,x,y):
	total = 0
	for i in range(len(x)):
		total+=((x[i]*w[i])+(y[i]*w[i]))#+(z[i]*w[i]))
	return total

def load_model_and_test(model,xtest,ytest,test_results):
	f = open(model,"rb")
	reg = pickle.load(f)
	pred = reg.predict(xtest)
	mbe = np.mean(ytest-pred)
	mape = mean_absolute_percentage_error(ytest,pred)
	print(ytest[:5])
	print(pred[:5])
#	print(pred[:5])
#	print(mean_squared_error(ytest,pred))
	test_results.loc[(len(test_results.index))] = {"Model":model,"MSE":mean_squared_error(ytest,pred),"MAE":mean_absolute_error(ytest,pred),"MBE":mbe}
	return pred

def main(params,accuracy_metrics):
	global xtrain,ytrain,xtest,ytest,model_name,dir
	global optimisers
	optimisers = {"Adam":keras.optimizers.Adam,"SGD":keras.optimizers.SGD,"RMSprop":keras.optimizers.RMSprop,"Adadelta":keras.optimizers.Adadelta,"Adagrad":keras.optimizers.Adagrad,"Adamax":keras.optimizers.Adamax,"Nadam":keras.optimizers.Nadam,"Ftrl":keras.optimizers.Ftrl}
	model_name = params["mode"]
	model_details = ""
	params["speech_labels"] = ["arousal","valence"]
	params["output_dir"] = params["results_dir"]+params["features"]+" + "+params["img"]+"/"
	if not os.path.isdir(params["output_dir"]):
		os.makedirs(params["output_dir"])
	dir = params["output_dir"]
	date_stamp = str(datetime.now())
	start_time = datetime.now()
	dataset = params["dataset"]
	forbidden_characters = [':','/',"\\",'?','<','>','|',"...",".."]
	for i in forbidden_characters:
		date_stamp = date_stamp.replace(i, '.')
	results_file = params["output_dir"]+dataset+"_output"+date_stamp+".txt"
	sys.stdout = open(results_file,'w')
	# gives a single float value
	print(f"CPU Percent: {psutil.cpu_percent()}")
	# you can convert that object to a dictionary 
	print(dict(psutil.virtual_memory()._asdict()))
	# you can have the percentage of used RAM
	print(f"Virtual Memory Percentage: {psutil.virtual_memory().percent}")
	# you can calculate percentage of available memory
	print(f"Percentage of Available Memory Used: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total}")
	last_char = params["results_dir"][len(params["results_dir"])-1]
	if last_char != "/":
		params["results_dir"] = params["results_dir"]+"/"
	print("Feature Extraction Method: "+params["features"])
	print("Image Type: "+params["images"])
	print()
	# print(params)
	# dataset_directory = os.getcwd()
	sys.stdout.flush()
	# test_results = pd.DataFrame(columns=("R2","MSE","MAE","MAPE"))
	# results = pd.DataFrame(columns=("Model","R2 Train","R2 Test","MAE","MSE","RMSE","MBE","MAPE"))
	# if params["img"] != "raw":
	fp = open(params["dataset_dir"]+f"Images {params['features']} {params['images']}.pickle","rb")
	speech_train_full = pickle.load(fp)
	fp.close()
	speech_train = []
	for s in speech_train_full:
		speech_train.append(s.get("data"))
	fp = (params["dataset_dir"]+f"{dataset} Sounds NA 5 labels.csv")
	speech_y = pd.read_csv(fp)
	speech_y = speech_y[params["labels"]]
	# else:
	# 	if params["features"] != "NA":
	# 		speech_train = pd.read_csv(params["dataset_dir"]+f"KEmo-Con_{params['features']}_raw_Speech_Data.csv",index_col=0)
	# 	else:
	# 		speech_train = pd.read_csv(params["dataset_dir"]+"IEMOCAP_All_Data.csv",index_col=0)
		# speech_y = speech_train[params['speech_labels']]
		# speech_y.drop(columns=["subject","seconds"],inplace=True)
		# speech_train.drop(columns=params['speech_labels'],inplace=True)
		# speech_train = speech_train.to_numpy()
		# speech_y = speech_y.to_numpy()
	speech_y = normalise_column_max_min(speech_y).round(3)
	# thirty_percent = int(np.shape(speech_train)[0]*0.3)
	speech_train,speech_y = shuffle(speech_train,speech_y,random_state=0)
	# print(f"Using 30% of the full dataset, {thirty_percent} samples total.")
	print("Dataset loaded.")
	print(np.shape(speech_y))
	print(np.shape(speech_train))
	print(f"Speech data sample: {speech_train[0]}")
	sys.stdout.flush()
	if params["model_type"] != "CNN" and len(np.shape(speech_train)) > 3:
		speech_train = np.reshape(speech_train,[np.shape(speech_train)[0],np.shape(speech_train)[1],-1])
	speech_train
	print("Training Beginning...")
	sys.stdout.flush()
	xtrain,xtest,ytrain,ytest = train_test_split(speech_train,speech_y,test_size=0.2,random_state=42)
	mae = []
	mae = 0
	mse = 1
	mbe = 1
	r2 = 0
	iqr=100
	if params["model_type"] == "CNN":
		r2,mae,mse,mbe = (cnn_regression(params))
	elif params["model_type"] == "Linear":
		r2,mae,mse,mbe = (multioutput_regression(params))
	elif params["model_type"] == "RNN":
		r2,mae,mse,mbe = (RNN_simple(params))
	elif params["model_type"] == "LSTM":
		r2,mae,mse,mbe = (Time_LSTM(params))
	elif params["model_type"] == "BiRNN":
		r2,mae,mse,mbe = (BiRNN(params))
	elif params["model_type"] == "BiLSTM":
		r2,mae,mse,mbe = (BiLSTM(params))
	elif params["model_type"] == "RF":
		r2,mae,mse,mbe = (random_forest(params))
	elif params["model_type"] == "DT":
		r2,mae,mse,mbe = (decision_tree(params))
	elif params["model_type"] == "KNN":
		r2,mae,mse,mbe = (nearest_neighbour(params))
	elif params["model_type"] == "MLP":
		r2,mae,mse,mbe = MLP_simple(params)
	stop_time = datetime.now()
	dur = stop_time-start_time
	# mt.stop()
	# rt.stop()
	print(f"Time taken: {dur}")
	sys.stdout.flush()
	print(f"CPU Percent: {psutil.cpu_percent()}")
	# you can convert that object to a dictionary 
	print(dict(psutil.virtual_memory()._asdict()))
	# you can have the percentage of used RAM
	print(f"Virtual Memory Percentage: {psutil.virtual_memory().percent}")
	# you can calculate percentage of available memory
	print(f"Percentage of Available Memory Used: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total}")
	sys.stdout.flush()
	current_process = psutil.Process()
	children = current_process.children(recursive=True)
	for child in children:
		print('Child pid is {}'.format(child.pid))
		os.killpg(child.pid, signal.SIGKILL)
	return r2,mae,mse,dur