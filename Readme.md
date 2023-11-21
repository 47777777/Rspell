# RSpell
Code for paper "RSpell: Retrieval augmented Framework for Domain Adaptive Chinese Spelling Check"

## Usage:
1. cd Code
   python tokenize_sequence_add.py
2. bash script.sh
3. cd Code
   python evaluate.py

## Data usage
Path: `csc_evaluation/builds/sim/domain`
- For zero-shot tasks, you should train on sighan2015_js(as activation) and evaluate on the sum*_js.txt file.
- For common tasks, the *.train file and the *.dev file are used to do training and do evaluating while *.test is adopted to do predicting.


[Model weights]
(https://github.com/Aopolin-Lv/ECSpell)
(https://drive.google.com/file/d/1HlfDbMpXR6YHiBuJS8s_K3ZKG6j0fvc5/view?usp=sharing)