Neural Machine Translation Codebase in Matlab 
====================================================================

Code to train state-of-the-art neural machine translation systems that can
handle very complex languages like Czech using hybrid word-character models described in our 
<a href="http://nlp.stanford.edu/pubs/luong2016acl_hybrid.pdf">ACL'16 paper</a>. 

This codebase can also train general attention-based models described in our
<a href="http://nlp.stanford.edu/pubs/emnlp15_attn.pdf">EMNLP'15 paper</a> and
has all the functionalities of our previous 
<a href="https://github.com/lmthang/nmt.matlab">nmt.matlab</a> codebase.

Why Matlab? It was a great learning experience for me to be able to derive by
hand all gradient formulations and implement everything from crash! 
Matlab supports GPU, so the code is also very fast.

## Features:
- Train hybrid word-character as well as general attention-based models.
- Beam-search decoder that can ensembles models including hybrid ones.
- Code to compute source word representations and evaluate on the word
  similarity tasks or do tsne plots.
- Code to compute sentence representations and rerank scores.

## Citations:
If you make use of this codebase in your research, please cite our paper
```
@inproceedings{luong2016acl_hybrid,
 author = {Luong, Minh-Thang  and  Manning, Christopher D.},
 title = {Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models},
 booktitle = {Association for Computational Linguistics (ACL)},
 address = {Berlin, Germany},
 month = {August},
 year = {2016}
}

```

- Thang Luong <lmthang@stanford.edu>, 2014, 2015, 2016

## Files

```
README.md       - this file
code/           - main Matlab code
  trainLSTM.m: train models
  testLSTM.m: decode models
  computeSentRepresentations.m: compute encoder representations.
  computeRerankScores.m: compute decoding scores.
data/           - toy data
scripts/        - scripts
```
## Core scripts 
- We provide an one-for-all script that performs all the preprocessing steps & train a translation model
```
1-prepare_and_train.sh <trainPrefix> <validPrefix> <testPrefix> <srcLang> <tgtLang> <wordVocabSize> <charVocabSize> <outDataDir> <outModelDir> [options]
  trainPrefix   expect train files trainPrefix.(srcLang|tgtLang)
  validPrefix   expect valid files validPrefix.(srcLang|tgtLang)
  testPrefix    expect test files testPrefix.(srcLang|tgtLang)
  srcLang     Source languague
  tgtLang     Target languague
  wordVocabSize   Word vocab size.
  charVocabSize   Character vocab size. If 0, run word-based models.
  outDataDir    Output data directory where we save preprocessed data
  outModelDir   Output model directory that we save during training
  options     Options to trainLSTM
```
The script is smart enough to check if preprocessed data files have been created in <outDataDir> so that we can reuse. When <charVocabSize> is greater than 0, we will train hybrid word-character models.

## Examples
# Training
- Process data & train a hybrid model:
```
./scripts/1-prepare_and_train.sh data/train.10k data/valid.100 data/test.100 de en 1000 50 data.hybrid.50 model.hybrid.w1000.c50
```

- We can also add options such as dropout (keep probability = 0.8) and use 2-layer character-level models as below:
```
./scripts/1-prepare_and_train.sh data/train.10k data/valid.100 data/test.100 de en 1000 50 data.hybrid.50 model.hybrid.w1000.c50.dropout0.8.charLayer2 "'dropout',0.8,'charNumLayers',2"
```

- To train regular attention-based sequence-to-sequence NMT:
```
./scripts/1-prepare_and_train.sh data/train.10k data/valid.100 data/test.100 de en 1000 0 data.1000 model.w1000
```

# Testing (Decoding)

## Miscellaneous
- Gradient checks:
```
./scripts/run_grad_checks.sh > output/grad_checks.txt 2>&1
```
Then compare with the provided grad check outputs data/grad_checks.txt. They should look similar.

- The Matlab code/ directory further divides into sub-directories:
```
  basic/: define basic functions like sigmoid, prime. It also has an efficient way to aggreate embeddings.
  layers/: we define various layers like attention, LSTM, etc. with forward and backprop code.
  misc/: things that we haven't categorized yet.
  preprocess/: deal with data.
  print/: print results, logs for debugging purposes.
  wordsim/: word similarity task 
```
