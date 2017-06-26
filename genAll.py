import re
import os
import sys
import glob	
import argparse
import string as st

def main(args):

	# Finding current directory.
	CURRENT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
	print(CURRENT_DIR)

	# Parsing current Mod and Variant of vanilla code based on current directory.
	re_mod = re.search('Mod([0-9])', CURRENT_DIR)
	mod_str = re_mod.group(0)

	re_var = re.search('Var([0-9])', CURRENT_DIR)
	var_str = re_var.group(0)

	# now we get the generation file name so that we can autoname the wav file with the generation number
	list_of_wavs = glob.glob(CURRENT_DIR + '/*.wav')
	latset_wav = max(list_of_wavs, key = os.path.getctime)
	print(latset_wav) # DEBUG
	gen_str = re.search('mod([0-9]+)var([0-9]+)gen([0-9]+)', latset_wav).group(3)



	# Create script
	file = mod_str + var_str + 'Gen.sh'
	open(file, 'w+')

	# Write shell script
	newline = '\n'

	# Flags
	file.write("#!/bin/bash -l" + newline)
	file.write("#$ -l h_rt=" + str(args.max_time_hours) + ":00:00" + newline)
	file.write("#$ -m beas ")
	file.write("#$ -N " + mod_str + var_str + gen_str + newline)
	file.write("#$ -pe omp " + str(args.num_cpus) + newline)
	file.write("#$ -l mem_per_core=" + str(args.mem_total) + newline)
	file.write("#$ -l gpus=" + str(args.num_gpus / args.cores) + newline)
	file.write("#$ -l gpu_c=" + str(args.gpu_c) + newline)

	file.write(newline + newline)

	# Load necessary modules
	file.write("module load cuda/8.0" + newline)
	file.write("module load cudnn/5.1" + newline)
	file.write("module load python/3.5.1" + newline)
	file.write("module load tensorflow/r1.1_python-3.5.1" + newline)

	file.write(newline)

	# Generation command
	file.write("python /projectnb/textconv/WaveNet/Code/tensorflow-wavenet/generate.py \ " + newline)
	file.write("\t\t--logdir=/projectnb/textconv/WaveNet/Models/" + mod_str + "/" + var_str + gen_str + " \ " + newline)
	file.write("\t\t--samples=" + str(args.samples) + "\ " + newline)
	
	checkpoint = "model.ckpt-{}".format(args.ckpt)
	file.write("\t\t--wavenet_params=/projectnb/textconv/WaveNet/Models/" + mod_str + var_str + ".json \ " + newline)
	file.write("\t\t/projectnb/textconv/WaveNet/Models/" + mod_str + "/" + var_str + "/Logs/" + checkpoint + " \ " + newline)

	file.write(newline)

	file.close()


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

	parser.add_argument('-gc', '--gpu-c',
						help = "GPU compute capability. 6.0 for a Tesla P100. Defaults to 3.5 for a Tesla K40m.",
						type = float,
						dest = 'gpu_c',
						default = 3.5,
						required = False)

	parser.add_argument('-c', '--cores',
						help = "Total number of CPU cores to request for training. Default is 4 cores.",
						type = int,
						dest = 'num_cpus',
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
						required = False)

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
