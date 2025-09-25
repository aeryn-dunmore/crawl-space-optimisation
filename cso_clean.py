import pandas as pd 
import sys
from datetime import datetime
import numpy as np
import random
import importlib.util

global params,bounds
params = {}
bounds = {}

def generate_start_point(points_list,these_bounds,int_val):
	start_point = np.clip(random.choice(points_list),*these_bounds)
	if int_val:
		start_point = int(start_point.round(0))
	return start_point
	

def generate_new_point(last_point,rate_of_change,these_bounds,int_val):
	new_point =  np.clip(abs(last_point*(1+(random.choice((-rate_of_change,rate_of_change))/100.0))),*these_bounds)
	if int_val:
		new_point = int(new_point.round(0))
	return new_point


def check_current_best(current_point,best_point):
	c = 0
	best = False
	for c in range(len(current_point)):
		if current_point[c] > best_point[c]:
			return True
		elif current_point[c] < best_point[c] and not best:
			return False
	return best


def continuous_recursive_best(current,best,loop,equal):
	global params
	# print(f"loop {loop}, current: {current[loop]}, best: {best[loop]}, equal: {equal}")
	if loop == 0:
		if ((current[loop] > best[loop]) and not params["up"][loop]) or ((current[loop] < best[loop]) and params["up"][loop]):
			return "worse"
		elif (current[loop] == best[loop]):
			return "equal"
		elif equal == "best" and (current[loop] == best[loop]):
			return "equal"
		elif ((current[loop] < best[loop]) and not params["up"][loop]) or ((current[loop] > best[loop]) and params["up"][loop]):
			return "best"
		else:
			return "equal"
	else:
		equal = continuous_recursive_best(current,best,loop-1,equal)
		if equal == "best":
			return "best"
		elif (((current[loop] > best[loop]) and not params["up"][loop]) or ((current[loop] < best[loop]) and params["up"][loop])) and (equal == "equal"):
			return "worse"
		elif (current[loop] == best[loop]) and (equal == ("equal" or "")):
			return "equal"
		elif ((current[loop] < best[loop]) and not params["up"][loop]) or ((current[loop] > best[loop]) and params["up"][loop]):
			return "best"
			equal = recursive_best(current,best,loop-1,"worse")
		# else:
		#     equal = recursive_best(current,best,loop-1,"equal")
		# print(f"loop {loop}, current: {current[loop]}, best: {best[loop]}, equal: {equal}")
		return equal
 

def categorical_recursive_best(current,best,loop,equal):
	# print(f"loop {loop}, current: {current[loop]}, best: {best[loop]}, equal: {equal}")
	if loop == 0:
		if (current[loop] > best[loop]):
			return "best"
		elif (current[loop] == best[loop]):
			return "equal"
		elif equal == "best" and (current[loop] == best[loop]):
			return "best"
		elif current[loop] < best[loop]:
			return "worse"
		else:
			return "equal"
	else:
		equal = categorical_recursive_best(current,best,loop-1,equal)
		if equal == "best":
			return "best"
		elif (current[loop] > best[loop]) and (equal == "equal"):
			return "best"
		elif (current[loop] == best[loop]) and (equal == ("equal" or "")):
			return "equal"
		elif current[loop] < best[loop]:
			return "worse"
			equal = recursive_best(current,best,loop-1,"worse")
		# else:
		#     equal = recursive_best(current,best,loop-1,"equal")
		# print(f"loop {loop}, current: {current[loop]}, best: {best[loop]}, equal: {equal}")
		return equal

def iter(iteration,current_point=None,val_iteration=0):
	global best_point,params,best,since_best,generate_new,params_to_opt,accuracy_metrics,iterations,roc,stop_point
	global specify_start, convergence_value, bounds_list, points_list, int_val, all_trials, outfile,csv_file,first,val_options, vals
	if iteration == 0:
		params_to_opt = params["models_req"][params["model_type"]]
	new_point = {}
	folder_name = ""
	for v in vals:
		folder_name = folder_name+str(params[v])+"_"
	params["results_dir"] = params["output_dir"]+f"results/{folder_name}{iteration}_{val_iteration}/"
	if (not generate_new) and (not first) and (current_point is not None):
		for p in params_to_opt:
			new_point[p] = generate_new_point(current_point[p],roc,bounds_list[p],int_val[p])
	else:
		for p in params_to_opt:
			if specify_start and first:
				new_point[p] = current_point[p]
			elif specify_start:
				new_point[p] = generate_new_point(best_point[p],roc,bounds_list[p],int_val[p])
			else:
				new_point[p] = generate_start_point(points_list[p],bounds_list[p],int_val[p])
		if first:
			best_point = new_point
			best = []
	for p in params_to_opt:
		params[p] = new_point[p]
	results = []
	is_better = ""
	network = importlib.util.spec_from_file_location(params["network"], params["network_script"])
	called_script = importlib.util.module_from_spec(network)
	network.loader.exec_module(called_script)
	# Call function with the arguments
	results = called_script.main(params,accuracy_metrics)
	measures = results[params["measures"]]
	if params["categorical"] and not first:
		is_better = categorical_recursive_best(measures,best,0,"")
	elif not first:
		is_better = continuous_recursive_best(measures,best,0,"")
	else:
		is_better = "best"
	sys.stdout = outfile
	print(f"\nBest: {best}\nCurrent: {results}")
	print(f"Eval results: {is_better}")
	sys.stdout.flush()
	if is_better == "best":
		best = results
		best_point = current_point
		since_best = 0
		generate_new = False
	else:
		if ((since_best%stop_point == 0) and (since_best != 0)):
			generate_new = True
			since_best+=1
		else:
			generate_new = False
			since_best+=1
	# out_cols = list(params_to_opt).extend(accuracy_metrics)
	# print(out_cols)
	r = 0
	out_vals = {}
	for v in vals:
		out_vals[v] = str(params[v])
	for n in params_to_opt:
		out_vals[n] = new_point[n]
	# print(out_vals)
	r = 0
	for a in accuracy_metrics:
		out_vals[a] = results[r]
		r+=1
	# print(out_vals)
	out_df = pd.DataFrame(index=[iteration],data=out_vals)
	if first:
		header = True
	else:
		header = False 
	out_df.to_csv(csv_file,header=header)
	outfile.write(out_df.to_string())
	outfile.flush()
	all_trials+=1
	return new_point


def recursive_loop(iteration,current_point,loop_number):
	global vals,params,base_loop, val_options,val_iters,params_to_opt,accuracy_metrics,iterations,roc
	global convergence_value, bounds_list, points_list, int_val, all_trials, outfile,csv_file
	if loop_number > 0:
		for v in val_options[vals[loop_number]]:
			params[vals[loop_number]] = v
			print(f"{vals[loop_number]}: {v}")
			recursive_loop(iteration,current_point,(loop_number-1))
			# val_iters[vals[loop_number]]+=1
	else:
		val_eval = vals[0]
		for v in range(len(val_options[val_eval])):
			for i in range(iterations):
				params[vals[loop_number]] = val_options[val_eval][v]
				print(f"{vals[loop_number]}: {params[vals[loop_number]]}")
				global first
				# params_to_opt = params["model_req"][vals["model"]]
				iter(i,current_point,iteration)
				if first:
					first = False

def main(param):
	global vals, val_options,val_iters,params_to_opt,stop_point,accuracy_metrics,iterations,roc,convergence_value, bounds_list, points_list, int_val, all_trials, outfile,csv_file
	global params,generate_new, since_best,circling,specify_start
	params = param
	specify_start = params["specify_start"]
	since_best = 0
	generate_new = False
	vals = params["iterative_params"] # Which values/settings are we going to try the different combinations of?
	val_options = params["iterative_options"] # Get all values for the combinations.
	val_iters = {}
	for v in vals:
		val_iters[v] = 0
	params_to_opt = params["to_optimise"] # Which hyperparameters are intended for optimisation?
	accuracy_metrics = params["metrics"] # What are the metrics the iterations will be evaluated on?
	iterations = params["iterations"] # How many optimisation iterations should be performed per combination?
	roc = params["rate_of_change"] # What is the percent rate of change for the parameters?
	convergence_value = params["convergence"] # Get the value at which convergence is considered to be achieved.
	bounds_list = params["bounds_list"] # Get the boundary values for each parameter for optimisation.
	points_list = params["points_list"] # Get the range of starting points for each parameter.
	int_val = params["int_val"] # Get the boolean value for whether the value is an integer.
	stop_point = params["tries"]
	all_trials = 0 
	date_stamp = datetime.now()
	outfile = open(params["output_dir"]+f"CSO output {date_stamp}.txt","w") # Create a file for all program output.
	csv_file = open(params["output_dir"]+f"CSO results {date_stamp}.csv","w") # Create a csv file for tabulated results.
	outfile.write(F"Crawl Space Optimisation - {date_stamp}\n")
	outfile.flush()
	params_to_opt = {"lr","neurons","epochs","gamma","max_depth","min_cweight","subsample","n_estimators","colsamples"}
	sys.stdout = outfile
	sys.stdout.flush()
	global first
	first = True
	best_point = None
	try:
		for i in range(len(val_options[vals[0]])):
			global base_loop
			base_loop = 0
			recursive_loop(i,None,len(vals)-1)
	except Exception as e:
		print(e)
	
