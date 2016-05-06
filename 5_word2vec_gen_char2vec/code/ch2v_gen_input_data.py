# generate the input file for generating character2vector

import re

fpath_list = ['data/task/task-correct.csv', 'data/task/task-error.csv']
# fpath_list = ['data/nontask/nontask-correct.csv', 'data/task/task-correct.csv']
output_fpath = 'output/ch2v_corpus_output'

output_str = ''

for fpath in fpath_list:
	with open(fpath) as input_f:
		line = input_f.readline()
		# line = input_f.readline().replace(" ", "").strip('\n')[:-1]
		while line != '':
			line = re.sub('[^0-9a-zA-Z]+', '', line).lower()
			line = (" ".join(line))
			output_str = output_str + line + ' '
			line = input_f.readline()
			# line = input_f.readline().replace(" ", "").strip('\n')[:-1]

with open(output_fpath, 'w') as output_f:
	output_f.write(output_str.strip())