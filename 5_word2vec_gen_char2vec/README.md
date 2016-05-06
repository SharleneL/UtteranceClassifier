# Files and folders
- `code/`: this folder contains all the code, including the word2vec open source provided by Google. The code can be found [here](https://code.google.com/archive/p/word2vec/).
- `data/`: this folder contains the datasets used in the experiments, including task/nontask correct/error datasets.
- `output/`: this folder will save all the output files generated during the experiment, including the input file generated in Step1 (shown below) and the character2vector model generated in Step2 (shown below).

# Running instructions
This is the instruction to generate `character2vector`

- Step1: Generate corpus file (a single line of **error sentence** characters separated by **spaces**, eg.: "I l o v e y o u"):
	- Under `code/` folder, run `python ch2v_gen_input_data.py`, will generate the needed corpus file under `output/` folder
- Step2: Generate character2vector model file
	- Under `code/` file, run:
		1. `gcc word2vec.c -o word2vec`
		2. `time ./word2vec -train ../output/ch2v_corpus_output -output ../output/ch2v.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15` 
	Then we can get `ch2v.bin`  under `output` folder, which is the character2vector model we need, and its file path will be set in `tools.py` in `skipgram_gen_word_vectors` experiment.
	- To test: under `code/` folder, run `gcc -o distance distance.c`, and `./distance ../output/ch2v.bin`. Then you can input whatever character you wanna test distance.
	- To tune the command `time`, change the following parameters:
		- `-size` desired vector dimensionality
		- `-window` the size of the context window for either the Skip-Gram or the Continuous Bag-of-Words model
		- `-negative` training algorithm: hierarchical softmax and / or negative sampling
		- `-sample` threshold for downsampling the frequent words 
		- `-threads` number of threads to use
		- `-binary` the format of the output word vector file (text or binary)