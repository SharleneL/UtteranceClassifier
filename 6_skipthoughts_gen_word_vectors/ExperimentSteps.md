# Skipgram - Instruction to Generate Word Vectors

## Prepare training & testing files
- Make training & testing formatted files:  format is **label \s text**
- Put in `thoughtsVec/data`: testing data `train_correct` and `test_correct`
- Change filepaths in `thoughtsVec/eval_trec.py`: change the train & testing data path in 'with open...' line

## 0. Connect to server
- The model is trained on server, thus needs to shh to the server
- Command: `ssh -p 22222 kyusonglee@141.223.162.18`
- passwd:12345678

## 1. Generate character2vector .bin file

### 1. Generate `character2vector` bin file
- Check out the details in experiment folder `word2vec_gen_char2vec/`

### 2. Set **path\_to\_model** in `tools.py`
- location is in  `~/temp/skip-thoughts/training/model.npz`

### 3. Set **path\_to\_dic** in `tools.py`

## 2. Generate input file (word dictionary to be converted into vectors)
- Run `0417_gen_ch2vec_input_data.py`, will generate a dictionary for task-correct & task-error data. In this python file, the first section of code generates a dictionary with all words. The second section generates a dictionary only contains the words which has both correct and error versions from the task-correct & task-error data(AKA, if the word appears in both datasets in the correct form, it would not be added to the dic).
- The generated file contains each word in a line, and each word separated by spaces. Next step we will generate the vectors for all these words.

## 3. Use trained model to generate vectors
- On server,  put the input file we got in last step under `~/temp/skip-thoughts/training/data/`
- Under `~/temp/skip-thoughts/training/` folder, run `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python encode.py`, then under the `/data` folder we can get the vector file. Each line of this file represents one line of word in the input file.

## 4. Plot
- Use the output file of `Generate Input Words` section, and the output file of `Generate the Vectors` section.
- Input the contents of the above 2 files [here](http://cs.stanford.edu/people/karpathy/tsnejs/csvdemo.html), then run and generate the plotting result.
- Plotting: [tool link](http://lvdmaaten.github.io/tsne/#implementations)