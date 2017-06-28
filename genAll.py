import re
import os
import sys
import glob
import argparse
import string as st

def main(args):

	# FIND CURRENT DIR
	CURRENT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))

	# CURRENT MOD AND VARIANT OF MODEL BASED ON CURRENT DIR
	re_mod = re.search('Mod([0-9])', CURRENT_DIR)
	mod_str = re_mod.group(0)
	mod_num = re_mod.group(1)

	# VAR STRING AND NUMBER
	re_var = re.search('Var([0-9])', CURRENT_DIR)
	if re_var is None:
		var_str = re.search('Mod([0-9]+)/([a-zA-Z0-9]+/?)', CURRENT_DIR).group(2)
		var_num = None
	else:
		var_str = re_var.group(0)
		var_num = re_var.group(1)


	# ALL WAV FILES IN CWD AND FROM THOSE, THE LATEST
	list_of_wavs = glob.glob(CURRENT_DIR + '/*.wav')
	latest_wav = max(list_of_wavs, key = os.path.getctime)
	
	# FROM THAT, CURRENT LATEST WAV FILE
	if var_num is not None:
		gen_match_str = 'M([0-9]+)V([0-9]+)G([0-9]+)'
		gen_str = re.search(gen_match_str, latest_wav).group(3)
	else:
		gen_match_str = "G([0-9]+)".format(var_str)
		gen_str = re.search(gen_match_str, latest_wav).group(1)
	
	# NUMBER OF NEXT GENERATION FILE
	if gen_str is not None:
		gen_num = int(gen_str) + 1 if args.gen_num is None else args.gen_num
	else:
		gen_num = 1 if args.gen_num is None else args.gen_num

	# NOW MAKE THE NAME OF NEXT GENERATION FILE
	if var_num is not None:
		gen_file_name = "M{}V{}G{}".format(mod_num, var_num, gen_num)
	else:
		gen_file_name = "M{}{}G{}".format(mod_num, var_str, gen_num)


	# NOW FIND OUT THE LATEST CHECKPOINT NUMBER
	os.chdir(os.path.join(CURRENT_DIR, 'Logs'))
	list_of_ckpt = glob.glob('*.ckpt-*')
	latest_ckpt_file = max(list_of_ckpt, key = os.path.getctime)
	os.chdir(CURRENT_DIR)
	latest_ckpt_number = re.search('ckpt-([0-9]+)', latest_ckpt_file).group(1)

	# DECIDE WICH ONE TO USE AND SET CKPT NUMBER
	ckpt = args.ckpt if args.ckpt is not None else latest_ckpt_number



	# NAME OTHER REQUIRED FILE ACCODINGLY
	file_name = mod_str + var_str + 'Gen.sh'
	params_file_name = mod_str + var_str + ".json"
	params_file_name = params_file_name.lower()



	# OPEN FILE FOR WRITING
	file = open(file_name, 'w+')

	# WRITE SHELL SCRIPT
	# Bash shell declaration
	file.write("#!/bin/bash -l\n")

	# Welcome message
	file.write("# Training Script Autogenerated by UniversalScripter : https://github.com/thakkarV/UniversalScripter\n")
	file.write("# To be used for training WaveNet models on the BU SCC\n\n")

	# Set project name in the SCC
	file.write("# Project name\n")
	name_str = "#$ -P {} \n\n".format(mod_str + var_str + gen_str if args.job_name is None else args.job_name)
	file.write(name_str)

	# Max time to run generation
	file.write("# Maximum time to run generation job\n")
	time_str = "#$ -l h_rt={}:00:00\n\n".format(args.max_time_hours)
	file.write(time_str)	

	# Send an email when job begins and ends, and if job is aborted or suspended
	file.write("# Send an email when the job begins and ends, and if the job is suspended or aborted\n")
	file.write("#$ -m beas \n\n")

	# Set job name in the SCC
	file.write("# Job name\n")
	name_str = "#$ -N {}\n\n".format(mod_str + var_str + 'Gen' + str(gen_num) if args.job_name is None else args._job_name)
	file.write(name_str)

	# Number of CPU cores to request from the SCC
	file.write("# Number of CPU num_cpus\n")
	cpu_str = "#$ -pe omp {}\n\n".format(args.num_cpus)
	file.write(cpu_str)

	# Total memory to request from the SCC
	file.write("# Total job memory\n")
	mem_str = "#$ -l mem_total={}G \n\n".format(args.mem_total)
	file.write(mem_str)

	# Number of GPUs to request from the SCC
	file.write("# Number of GPUS\n")
	file.write("#$ -l gpus=" + str(args.num_gpus / args.num_cpus) + "\n\n")

	# Minimum capability of GPU (3.5 for Tesla K40m, 6.0 for Tesla P100...)
	file.write("# GPU capability\n")
	gpuc_str = "#$ -l gpu_c={}\n".format(args.gpu_c)
	file.write(gpuc_str)

	file.write("\n")

	# Load necessary modules in the SCC
	file.write("# Load necessary modules\n")
	file.write("module load cuda/8.0\n")
	file.write("module load cudnn/5.1\n")
	file.write("module load python/2.7.13\n")
	file.write("module load tensorflow/r1.0_python-2.7.13\n")

	file.write("\n")

	# Generation command - python file run + flags
	file.write("# Generation command\n")
	file.write("python /projectnb/textconv/WaveNet/Code/tensorflow-wavenet/generate.py \\\n")


	# output path of the wav file
	wav_out_path_str = "\t\t--wav_out_path=/projectnb/textconv/WaveNet/Models/{}/{}/{}.wav \\\n" \
						.format(mod_str, var_str, gen_file_name)
	file.write(wav_out_path_str)


	# samples command
	samples_str = "\t\t--samples={} \\\n".format(args.samples)
	file.write(samples_str)
	

	# params file command
	params_str = "\t\t--wavenet_params=/projectnb/textconv/WaveNet/Models/{}/{}/{} \\\n" \
				.format(mod_str, var_str, params_file_name)
	file.write(params_str)


	# checkpoint command
	ckpt_str = "\t\t/projectnb/textconv/WaveNet/Models/{}/{}/Logs/model.ckpt-{} \n\n" \
				.format(mod_str, var_str, ckpt)
	file.write(ckpt_str)


	# all done writing
	file.close()

	# execute if required
	if args.execute:
		os_command = "qsub " + file_name
		os.system(os_command)

if __name__ == '__main__':

	# Create arguments on command line. Name of job and checkpoint number are required.
	parser = argparse.ArgumentParser(prog = "UniversalTrainer",
	                                description = "Automatic shell script generator for training and generating from WaveNet.")


	parser.add_argument('-gp', '--gpus',
						help = "Total number of GPUs to request for training. Default is one.",
						type = int,
						dest = 'num_gpus',
						default = 1,
						required = False)

	parser.add_argument('-gc', '--gpu_c',
						help = "GPU compute capability. 6.0 for a Tesla P100. Defaults to 3.5 for a Tesla K40m.",
						type = float,
						dest = 'gpu_c',
						default = 3.5,
						required = False)

	parser.add_argument('-c', '--cores',
						help = "Total number of CPU cores to request for training. Default is 4 num_cpus.",
						type = int,
						dest = 'num_cpus',
						default = 4,
						required = False)

	parser.add_argument('-mt', '--mem_total',
						help = "Total amount of RAM to request for training. Default is 32G.",
						type = int,
						dest = 'mem_total',
						default = 32,
						required = False)

	parser.add_argument('-t', '--time',
						help = "Maximum number of hours to train for. Default is 1 week.",
						type = int,
						dest = 'max_time_hours',
						default = 168,
						required = False)

	parser.add_argument('-n', '--name',
						help = "Name of training job.",
						type = str,
						dest = 'job_name',
						default = None,
						required = False)

	parser.add_argument('-p', '--project-name',
						help = "Name of SCC project.",
						type = str,
						dest = 'proj_name',
						default = 'textconv',
						required = False)

	parser.add_argument('-ck', '--ckpt',
						help = "Number of the checkpoint file to generate from. Uses the latest automatically",
						type = int,
						dest = 'ckpt',
						default = None,
						required = False)

	parser.add_argument('-s', '--samples',
						help = "Number of audio samples to generate. 16000 samples corresponds to 1 second of raw audio. Default is 16000.",
						type = int,
						dest = 'samples',
						default = 80000,
						required = False)

	parser.add_argument('-e', '--execute',
						help = "QSUB the generated automatically file if this flag is passed.",
						action = 'store_true',
						required = False)

	parser.add_argument('-ng', '--num_gen',
						help = "Number generation from this model.",
						type = int,
						dest = 'gen_num',
						default = None,
						required = False)

	args = parser.parse_args()

	# Run main to create script
	main(args)
