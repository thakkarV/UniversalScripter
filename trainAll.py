import re
import os
import sys
import argparse

def main(args):
	CURRENT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))

	re_mod = re.search('/Mod([0-9]+)/', CURRENT_DIR)
	print(re_mod)

	print(CURRENT_DIR)



if __name__ == '__main__':
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
						help = "Total amount of RAM to request for training. Default is 32G.",
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

	parser.add_argument('-n', '--name',
						help = "Name of training job.",
						type = str,
						nargs = 1,
						dest = 'job_name',
						default = None,
						required = True)

	args = parser.parse_args()

	print(args.job_name)
	main(args)
