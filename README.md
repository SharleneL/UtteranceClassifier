# UtteranceClassifier

Author: Shalin Luo | Jan 2016 - May 2016

## Introduction
This repository contains the experiments for the research on spoken dialogue system's utterance classifier. Each folder is one set of experiments, and contains:

- Code
- Experiment data
- A `README.md` documentation file

Following the README documentation under each folder, the experiments can be run.


## Contents
- `1_basic_classifiers`: Experiments to compare different Scikit-learn machine learning models' binary classification accuracy. Using BOW feature.
- `2_bow_nchargram`: Experiments to compare different scikit-learn machine learning models' binary classification accuracy. Using n-gram word and n-gram character features.
- `3_keras`: Experiments to compare logistic regression and deep learning models' multi-classification accuracy. Using correct/error dataset, BOW & trigram character feature, scikit-learn logistic regression model, Keras deep learning library.
- `4_skipthoughts_train`: Experiments to train a modified skip-thoughts model.
- `5_word2vec_gen_char2vec`: Experiments to build a character2vector model by modifying Google word2vec model.
- `6_skipthoughts_gen_word_vectors`: Experiments to generate word embeddings using skipthoughts model and character embeddings.
- `7_spell_correction`: Experiments to fix spelling errors by building a dictionary.
- `8_wordvec2sentvec_lr`: Experiments to combine word embeddings into sentence embeddings by:
	1. Average 
	2. Consine similarity statistics
	3. Skip-thoughts 
	
	Also includes a logistic regression classifier for the sentence embeddings.
