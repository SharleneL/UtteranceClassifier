# Introduction
This experiment uses [birkbeck spelling error corpus](http://www.dcs.bbk.ac.uk/~ROGER/missp.dat) to:
 1. Analyze a data set's error spelling words in the sentences
 2. Generate corrected sentences; save to output file
 
# Folders & files
 - `code/`: contains all code
    - `main.py`: contains the main function of the program
    - `helper.py`: contains helper functions
 - `data/`: contains all input data
    - `birkbeck.txt`: the birkbeck spelling error corpus; containing all correct and possible error words
 - `output/`: used for saving the corrected sentences files
 
# Running Instruction
 - Open `main.py`, set `dic_fpath` variable as the filepath of birkbeck corpus data; set `data_dir` as the **folder** containing all the input data to be analyzed
 - Under `code/` folder, run `python main.py`. Analysis of the input data will be printed; corrected files will be saved under `output/` folder.