# generate the input file for generating character2vector (each line is a space-separated chars)

import re

fpath_list = ['../data/utterances/task/task-correct.csv', '../data/utterances/task/task-error.csv', '../data/utterances/nontask/nontask-correct.csv', '../data/utterances/nontask/nontask-error.csv', '../data/chat/chatbot.txt']
output_fpath = '../output/ch2v_spaced_word_list.output'  # the dictionary to be converted to vectors

output_str = ''

# ----- # CODE TO GENERATE ALL WORDS DICTIONARY | START # ----- #
word_set = set()
for fpath in fpath_list:
    input_f = open(fpath)
    lines = input_f.readlines()
    input_f.close()

    for line in lines:
        line = line.strip()
        if line != '':
            line = re.sub('[^0-9a-zA-Z\s]+', '', line).lower()
            for word in line.split():
                word_set.add(word)
# ----- # CODE TO GENERATE ALL WORDS DICTIONARY | END # ----- #


# ----- # CODE TO GENERATE <correct, error> PAIRED DICTIONARY | START # ----- #
# correct_fpath = fpath_list[0]
# error_fpath = fpath_list[1]

# correct_inputf = open(correct_fpath)
# error_inputf = open(error_fpath)
# correct_line_list =  correct_inputf.readlines()
# error_line_list =  error_inputf.readlines()
# correct_inputf.close()
# error_inputf.close()

# word_set = set()
# for correct_line, error_line in zip(correct_line_list, error_line_list):
# 	correct_line = correct_line.strip()
# 	error_line = error_line.strip()
# 	if correct_line == '' or error_line == '':
# 		continue
# 	correct_line = re.sub('[^0-9a-zA-Z\s]+', '', correct_line).lower()
# 	error_line = re.sub('[^0-9a-zA-Z\s]+', '', error_line).lower()
# 	correct_wlist = correct_line.split()
# 	error_wlist = error_line.split()
# 	for correct_word, error_word in zip(correct_wlist, error_wlist):
# 		if correct_word != error_word:
# 			word_set.add(correct_word)
# 			word_set.add(error_word)
# ----- # CODE TO GENERATE <correct, error> PAIRED DICTIONARY | END # ----- #

# ----- # OUTPUT | START # ----- # 
with open(output_fpath, 'a') as output_f:
    for word in word_set:
        output_str = (" ".join(word)) + '\n'
        output_f.write(output_str)
# ----- # OUTPUT | END # ----- # 