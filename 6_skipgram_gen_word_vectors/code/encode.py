# Author: Shalin Luo
# Apr 14, 2016
# File to loading model from tools.py and encode new sentence

import tools

input_fpath = 'data/ch2v_list'
output_fpath = 'data/vector_output'
embed_map = tools.load_googlenews_vectors()
model = tools.load_model(embed_map)
# X = ['l o v e ', 'r u n ', 'l i v e', 'e n d']
word_list = []
with open(input_fpath) as f:
    word = f.readline().strip()
    while word != '':
        word_list.append(word)
        word = f.readline().strip()
vector_list = tools.encode(model, word_list)

# output
with open(output_fpath, 'a') as f_output:
    for vector in vector_list:
        output_str = (" ".join(str(elem) for elem in vector)) + '\n'
        f_output.write(output_str)
print '*****END*****'