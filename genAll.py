import re
import os
import sys
import argparse
import string as st

def main(args):

	# Finding current directory.
	CURRENT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))

	# Parsing current Mod and Variant of vanilla code based on current directory.
	re_mod = re.search('Mod([0-9])', CURRENT_DIR)
	mod = re_mod.group(0)

	re_var = re.search('Var([0-9])', CURRENT_DIR)
	var = re_var.group(0)

	# Create script
	scr = 'open("' + mod + var + 'Gen.sh", "w+")'
	bash = eval(scr)

	# Write shell script

	# Flags
	bash.write("#!/bin/bash -l\n")
	bash.write("#$ -l h_rt=" + str(args.max_time_hours) + ":00:00\n")
	bash.write("#$ -m beas \n")
	bash.write("#$ -N " + args.job_name + "\n")
	bash.write("#$ -pe omp " + str(args.cores) + "\n")
	bash.write("#$ -l mem_per_core=" + str(args.mem_total) + "\n")
	bash.write("#$ -l gpus=" + str(args.num_gpus) + "\n")
	bash.write("#$ -l gpu_c=" + str(args.gpu_c) + "\n")

	bash.write("\n\n")

	# Load necessary modules
	bash.write("module load cuda/8.0\n")
	bash.write("module load cudnn/5.1\n")
	bash.write("module load python/3.5.1\n")
	bash.write("module load tensorflow/r1.1_python-3.5.1\n")

	bash.write("\n")

	# Generation command
	bash.write("python /projectnb/textconv/WaveNet/Vijay/tensorflow-wavenet/generate.py \ \n")
	bash.write("\t\t--logdir=/projectnb/textconv/WaveNet/Models/" + mod + "/" + var + " \ \n")
	bash.write("\t\t--samples=" + str(args.samples) + "\ \n")
	
	check = "model.ckpt-{}".format(args.ckpt)
	bash.write("\t\t--wavenet_params=/projectnb/textconv/WaveNet/Vijay/tensorflow-wavenet/ParamMods/" + mod.lower() + var.lower() + ".json \ \n")
	bash.write("\t\t/projectnb/textconv/WaveNet/Models/" + mod + "/" + var + "/Logs/" + check + " \ \n")

	bash.write("\n")

	bash.close()


if __name__ == '__main__':

	# Create arguments on command line. Name of job and checkpoint number are required.
	parser = argparse.ArgumentParser(prog = "UniversalTrainer",
	                                description = "Automatic shell script generator for training and generating from WaveNet.")


	parser.add_argument('-gp', '--gpus',
						help = "Total number of GPUs to request for training. Default is one.",
						type = int,
						dest = 'num_gpus',
						default = 0.25,
						required = False)

	parser.add_argument('-gc', '--gpu-c',
						help = "GPU compute capability. 6.0 for a Tesla P100. Defaults to 3.5 for a Tesla K40m.",
						type = float,
						dest = 'gpu_c',
						default = 3.5,
						required = False)

	parser.add_argument('-c', '--cores',
						help = "Total number of CPU cores to request for training. Default is 4 cores.",
						type = int,
						dest = 'cores',
						default = 4,
						required = False)

	parser.add_argument('-mt', '--mem-total',
						help = "Total amount of RAM to request for training. Default is 128G.",
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
						required = True)

	parser.add_argument('-ck', '--ckpt',
						help = "Number of the checkpoint file to generate from.",
						type = int,
						dest = 'ckpt',
						default = None,
						required = True)

	parser.add_argument('-s', '--samples',
						help = "Number of audio samples to generate. 16000 samples corresponds to 1 second of raw audio.",
						type = int,
						dest = 'samples',
						default = 16000,
						required = False)

	args = parser.parse_args()

	# Run main to create script
	main(args)