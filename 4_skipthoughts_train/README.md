# Introduction
The code is used for training a modified skip-thoughts model used for generating word embeddings. The original skip-thoughts training code can be found [here](https://github.com/ryankiros/skip-thoughts/tree/master/training).

# Running Instruction
The training steps are almost the same as **Step1 - Step3** in the skip-thoughts Github page [here](https://github.com/ryankiros/skip-thoughts/tree/master/training). Except the input, which should follow the following instruction (note: this folder does not contain the training data, it can be downloaded from the link below):

Use the same [BookCorpus](http://www.cs.toronto.edu/~mbweb/) to train the model; the input data should be preprocessed as lines of words, each line is a word, with each character separated by a space. Example:

```
h o w
a r e
y o u
```