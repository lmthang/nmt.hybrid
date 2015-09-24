Code to train Neural Machine Translation systems as described in our EMNLP paper
"Effective Approaches to Attention-based Neural Machine Translation".
https://aclweb.org/anthology/D/D15/D15-1166.pdf

Features:
(a) Multi-layer Long-Short Term Memory (LSTM) encoder-decoder models.
(b) Attentional mechanisms.
(c) Beam-search decoder.
(d) Dropout

If you make use of this code in your research, please cite our paper with
details in https://aclweb.org/anthology/D/D15/D15-1166.bib.

Thang Luong <lmthang@stanford.edu>, 2014, 2015
With contributions from:
  Hieu Pham <hyhieu@stanford.edu> -- beam-search decoder.

/***********/
/** Files **/
/**********/
README.TXT      - this file
code/           - main Matlab code
data/           - toy data
scripts/        - utility scripts

/***************/
/** Main code **/
/***************/
(a) Prepare data
./scripts/prepare_data.py --size <vocab_size> <input_text_file> <output_integer_file>
The script will first create a vocabulary of the specified size and convert words in the text file into integers. Words not in the vocab file are assignedan <unk> symbol. The vocab file is saved under the name <input_text_file>.vocab.<vocab_size>

We can reuse saved vocab file as follows:
./scripts/prepare_data.py --vocab_file <vocab_file> <input_text_file> <output_integer_file>

There's a script to run for train/valid/test files:
run_prepare_data.sh trainFile validFile testFile vocabSize outDir [verbose]

(b) Train an LSTM
run.sh  Train LSTM models
  trainPrefix   expect train files trainPrefix.(srcLang|tgtLang)
  validPrefix   expect valid files validPrefix.(srcLang|tgtLang)
  testPrefix    expect test files testPrefix.(srcLang|tgtLang)
  srcLang     Source languague
  tgtLang     Target languague
  srcVocabFile      Source vocab file
  tgtVocabFile      Target vocab file
  outDir      Output directory
  lstmSize    Dimension of source word vectors
  learningRate    Learning rate
  maxGradNorm   Max grad norm
  initRange   Number of features for source binary tree traversal
  batchSize   Number of sentences per minibatch, larger gives faster training time but worse results
  numEpoches    Number of training epochs
  logFreq     Compute validation perplexities after [logFreq] dots printed
  otherOptions    Other options to trainLSTM
The script run.sh will call the main LSTM training code at code/trainLSTM.m
See the trainLSTM.m code for more options, e.g., training multiple layers.

For otherOptions, you can put things like "'dropout',0.8,'posModel',1":
'embCPU',1: to reduce memory footprint by putting embedding matrix on the CPU and only load the needed part onto GPU.
'attnFunc',1: train with an attentional mechanism (values 0, 1, 2, 4)
'attnOpt',1: attention alignment function (values 0, 1, 2, 3)
'dropout',0.8: use dropout with dropout probability of 0.2
'softmaxFeedInput': feed previous top hidden state to the next time step.

/*********************/
/** Sample commands **/
/*********************/
(a) Prepare the data
./scripts/run_prepare_data.sh ./data/train.10k.en ./data/valid.100.en ./data/test.100.en 1000 ./data/id.1000
./scripts/run_prepare_data.sh ./data/train.10k.de ./data/valid.100.de ./data/test.100.de 1000 ./data/id.1000

(b) Train a bilingual LSTM model
export MATLAB=matlab
./scripts/run.sh ../data/id.1000/train.10k ../data/id.1000/valid.100 ../data/id.1000/test.100 de en ../data/train.10k.de.vocab.1000 ../data/train.10k.en.vocab.1000 ../output 100 0.1 5 0.1 128 10 1 "'isAssert',1,'isResume',0"

To run directly in Matlab, cd into code/ directory and run:
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 'logFreq', 1, 'isResume', 0)

(c) Grad check
trainLSTM('', '', '', '', '', '', '', '', 0, 'isGradCheck', 1)
trainLSTM('', '', '', '', '', '', '', '../output', 'isGradCheck', 1, 'assert', 1, 'numLayers', 2, 'initRange', 1.0, 'attnFunc', 0, 'softmaxDim', 0, 'posModel', 0, 'dropout', 1)

(d) Profiling
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 'logFreq', 1, 'isProfile', 1)

(e) Train a monolingual LSTM model
export MATLAB=matlab
./scripts/run.sh ../data/id.1000/train.10k ../data/id.1000/valid.100 ../data/id.1000/test.100 "" en "" ../data/train.10k.en.vocab.1000 ../monoOutput 100 0.1 5 0.1 128 10 1 "'isBi',0,'isAssert',1,'isResume',0"

To run directly in Matlab, cd into code/ directory and run:
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', '', 'en', '', '../data/train.10k.en.vocab.1000', '../monooutput', 'logFreq', 1,'isResume',0, 'isBi', 0)

(f) Train on PTB data:
trainLSTM('../data/ptb/id/ptb.train', '../data/ptb/id/ptb.valid', '../data/ptb/id/ptb.test', '', 'en', '', '../data/ptb/ptb.train.txt.vocab.10000', '../monooutput', 'logFreq', 10,'isBi',0,'lstmSize',200)

(g) Train hard attention model on posAll data:
* Prepare data
./scripts/compute_word_dict.py -f 5 -o 1 ./data/train.10k en de ./data/train.10k.f5 > ./data/train.10k.f5.stderr 2>&1 &

./scripts/generate_unk_parallel_data.py --no_eos --separate_output --dict ./data/train.10k.f5.en-de.dict --src_output_opt 1 --reverse_alignment --src_vocab_size 1000 --tgt_vocab_size 1000 ./data/train.10k en de ./data/posAll.rel/train 1
./scripts/generate_unk_parallel_data.py --no_eos --separate_output --dict ./data/train.10k.f5.en-de.dict --src_output_opt 1 --reverse_alignment --src_vocab ./data/train.10k.en.vocab.1000 --tgt_vocab ./data/train.10k.de.vocab.1000 ./data/valid.100 en de ./data/posAll.rel/valid 1
./scripts/generate_unk_parallel_data.py --no_eos --separate_output --dict ./data/train.10k.f5.en-de.dict --src_output_opt 1 --reverse_alignment --src_vocab ./data/train.10k.en.vocab.1000 --tgt_vocab ./data/train.10k.de.vocab.1000 ./data/test.100 en de ./data/posAll.rel/test 1

~/lstm/scripts/reverse.py ~/lstm/data/posAll.rel/test.id.en ~/lstm/data/posAll.rel/test.id.reversed.en
~/lstm/scripts/reverse.py ~/lstm/data/posAll.rel/valid.id.en ~/lstm/data/posAll.rel/valid.id.reversed.en
~/lstm/scripts/reverse.py ~/lstm/data/posAll.rel/train.id.en ~/lstm/data/posAll.rel/train.id.reversed.en

To generate data with absolute positions, use the flag --absolute. The directory posAll was generated with that flag.

* train
trainLSTM('../data/posAll.rel/train.id','../data/posAll.rel/valid.id','../data/posAll.rel/test.id','en','de','../data/posAll.rel/train.vocab.en','../data/posAll.rel/train.vocab.de','../output',0,'logFreq',1,'isClip',0,'numLayers',1,'attnFunc',4,'isResume',0, 'isReverse', 1)
(here we use attnFunc=4 since the data contains relative positions. For attnFunc 3, needs to use absolute positions)

trainLSTM('../data/posAll.abs/train.id','../data/posAll.abs/valid.id','../data/posAll.abs/test.id','en','de','../data/posAll.abs/train.vocab.en','../data/posAll.abs/train.vocab.de','../output',0,'logFreq',1,'isClip',0,'numLayers',1,'attnFunc',3,'isResume',0, 'isReverse', 1)

(h) Train attention-based models:
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 'logFreq', 1, 'isResume', 0, 'attnFunc', 1, 'isReverse', 1)

(i) Train with booststraped mono models:
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 'logFreq', 1, 'isResume', 0, 'monoFile', '../monoOutput/model.mat')

trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 'logFreq', 1, 'isResume', 0, 'monoFile', '../monoOutput/model.mat', 'decodeUpdateEpoch', 2, 'decodeUpdateOpt', 2)

(j) Train on depparse data:
trainLSTM('../data/depparse/id/train', '../data/depparse/id/dev.100', '../data/depparse/id/test.100', 'en', 'dep', '../data/depparse/train.en.vocab.50000', '../data/depparse/train.dep.vocab.50000', '../output', 'logFreq', 1, 'isResume', 0)

trainLSTM('../data/depparse/id/train', '../data/depparse/id/dev.100', '../data/depparse/id/test.100', 'en', 'dep', '../data/depparse/train.en.vocab.50000', '../data/depparse/train.dep.vocab.50000', '../output', 'logFreq', 1, 'isResume', 0,'numLayers',2,'attnFunc',4,'isReverse',1,'depParse',1,'depRootId',12,'depShiftId',4,'assert',1)

(k) Tie embeddings:
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'en', 'en', '../data/train.10k.en.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 'logFreq', 1, 'isResume', 0, 'tieEmb', 1)

(l) Feed softmax vec as input:
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'en', 'en', '../data/train.10k.en.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 'logFreq', 1, 'isResume', 0, 'softmaxFeedInput', 1)

(j) Decode:
./scripts/test.sh '../output/modelRecent.mat' 3 10 10 '../output/translations.txt'
testLSTM('../output/modelRecent.mat', 3, 10, 10, '../output/translations.txt'
