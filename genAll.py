import re
import os
import sys
import argparse
import subprocess

def main(args):

	# Finding current directory.
	CURRENT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))

	# Parsing current Mod and Variant of vanilla code based on current directory.
	re_mod = re.search('Mod([0-9])', CURRENT_DIR)
	
	re_var = re.search('Var([0-9])', CURRENT_DIR)

	# Create script
	scr = 'open("' + re_mod.group(0) + re_var.group(0) + 'Train.sh", "w+")'
	bash = eval(scr)

	# Create script
	'''#!/bin/bash -l
	#$ -P textconv
	#$ -l h_rt=168:00:00
	#$ -m beas
	#$ -N job_name
	#$ -pe omp 4
	#$ -l mem_per_core=32
	#$ -l gpus=0.25
	#$ -l gpu_c=3.5

	module load cuda/8.0
	module load cudnn/5.1
	module load python/3.5.1
	module load tensorflow/r1.1_python-3.5.1

	# avoiding changing directories by writing out the full paths to everything in the command line
	python /projectnb/textconv/WaveNet/Vijay/tensorflow-wavenet/train.py \
			--data_dir=args.data_dir \
			--logdir=? \
			--wavenet_params=/projectnb/textconv/WaveNet/Vijay/tensorflow-wavenet/ParamMods/mod1var1.json \
			--silence_threshold=0'''


if __name__ == '__main__':

	# Create arguments on command line. Name of job and checkpoint number are required.
	parser = argparse.ArgumentParser(prog = "UniversalTrainer",
	                                description = "Automatic shell script generator for training and generating from WaveNet.")


	parser.add_argument('-gp', '--gpus',
						help = "Total number of GPUs to request for training. Default is one.",
						type = int,
						nargs = 1,
						dest = 'num_gpus',
						default = 0.25,
						required = False)

	parser.add_argument('-gc', '--gpu-c',
						help = "GPU compute capability. 6.0 for a Tesla P100. Defaults to 3.5 for a Tesla K40m.",
						type = float,
						nargs = 1,
						dest = 'gpu_c',
						default = 3.5,
						required = False)

	parser.add_argument('-c', '--cores',
						help = "Total number of CPU cores to request for training. Default is 4 cores.",
						type = int,
						nargs = 1,
						dest = 'num_cpus',
						default = 4,
						required = False)

	parser.add_argument('-mt', '--mem-total',
						help = "Total amount of RAM to request for training. Default is 128G.",
						type = int,
						nargs = 1,
						dest = 'num_memtotal',
						default = 32,
						required = False)

	parser.add_argument('-t', '--time',
						help = "Maximum number of hours to train for. Default is 1 week.",
						type = int,
						nargs = 1,
						dest = 'max_time_hours',
						default = 168,
						required = False)

	parser.add_argument('-d', '-data',
						help = "Full path to dataset directory. Default is NCS House.",
						type = str,
						nargs = 1,
						dest = 'data_dir',
						default = '/projectnb/textconv/WaveNet/Datasets/NCS/House',
						required = False)

	parser.add_argument('-n', '--name',
						help = "Name of training job.",
						type = str,
						nargs = 1,
						dest = 'job_name',
						default = None,
						required = True)

	parser.add_argument('-ck', '--ckpt',
						help = "Number of the checkpoint file to generate from.",
						type = int,
						nargs = 1,
						dest = 'ckpt',
						default = None,
						required = True)

	args = parser.parse_args()

	# Run main to create script
	main(args)