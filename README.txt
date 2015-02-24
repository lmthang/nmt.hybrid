Code to train Long-Short Term Memory (LSTM) models

Thang Luong <lmthang@stanford.edu>, 2014, 2015

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
  baseIndex     Base index
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

For otherOptions, you can put things like
'embCPU',1: to reduce memory footprint by putting embedding matrix on the CPU and only load the needed part onto GPU.

/*********************/
/** Sample commands **/
/*********************/
(a) Prepare the data
./scripts/run_prepare_data.sh ./data/train.10k.en ./data/valid.100.en ./data/test.100.en 1000 ./data/id.1000
./scripts/run_prepare_data.sh ./data/train.10k.de ./data/valid.100.de ./data/test.100.de 1000 ./data/id.1000

(b) Train a bilingual LSTM model
export MATLAB=matlab
./scripts/run.sh ../data/id.1000/train.10k ../data/id.1000/valid.100 ../data/id.1000/test.100 de en ../data/train.10k.de.vocab.1000 ../data/train.10k.en.vocab.1000 ../output 0 100 0.1 5 0.1 128 10 1

To run directly in Matlab, cd into code/ directory and run:
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 0, 'logFreq', 1)
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 0, 'logFreq', 1, 'sortBatch', 1, 'shuffle', 1, 'isResume', 0)

(c) Grad check
trainLSTM('', '', '', '', '', '', '', '', 0, 'isGradCheck', 1)
trainLSTM('', '', '', '', '', '', '', '../output', 0, 'isGradCheck', 1, 'numLayers', 2, 'lstmOpt', 0, 'initRange', 10.0, 'attnFunc', 0, 'assert', 1, 'softmaxDim', 0, 'posModel', 0)

(d) Profiling
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 0, 'logFreq', 1, 'isProfile', 1)

(e) Train a monolingual LSTM model
export MATLAB=matlab
./scripts/run.sh ../data/id.1000/train.10k ../data/id.1000/valid.100 ../data/id.1000/test.100 "" en "" ../data/train.10k.en.vocab.1000 ../output 0 100 0.1 5 0.1 128 10 1

To run directly in Matlab, cd into code/ directory and run:
trainLSTM('../data/id.1000/train.10k', '../data/id.1000/valid.100', '../data/id.1000/test.100', '', 'en', '', '../data/train.10k.en.vocab.1000', '../output', 0, 'logFreq', 1)

(f) Train on PTB data:
trainLSTM('../data/ptb/id/ptb.train', '../data/ptb/id/ptb.valid', '../data/ptb/id/ptb.test', '', 'en', '', '../data/ptb/ptb.train.txt.vocab.10000', '../output', 0, 'logFreq', 10,'isBi',0,'lstmSize',200)

(g) Train on posAll data:
* Prepare data
./scripts/compute_word_dict.py -f 5 -o 1 ./data/train.10k en de ./data/train.10k.f5 > ./data/train.10k.f5.stderr 2>&1 &
./scripts/generate_unk_parallel_data.py --no_eos --separate_output --dict ./data/train.10k.f5.en-de.dict --src_output_opt 1 --reverse_alignment --src_vocab_size 1000 --tgt_vocab_size 1000 ./data/train.10k en de ./data/posAll/train 1
./scripts/generate_unk_parallel_data.py --no_eos --separate_output --dict ./data/train.10k.f5.en-de.dict --src_output_opt 1 --reverse_alignment --src_vocab ./data/train.10k.en.vocab.1000 --tgt_vocab ./data/train.10k.de.vocab.1000 ./data/valid.100 en de ./data/posAll/valid 1
./scripts/generate_unk_parallel_data.py --no_eos --separate_output --dict ./data/train.10k.f5.en-de.dict --src_output_opt 1 --reverse_alignment --src_vocab ./data/train.10k.en.vocab.1000 --tgt_vocab ./data/train.10k.de.vocab.1000 ./data/test.100 en de ./data/posAll/test 1
* train
trainLSTM('../data/posAll/train.id','../data/posAll/valid.id','../data/posAll/test.id','en','de','../data/posAll/train.vocab.en','../data/posAll/train.vocab.de','../output',0,'logFreq',1,'isClip',0,'numLayers',1,'posModel',1)

(h) Decode:
testLSTM('../output/modelRecent.mat', 3, 10, 10, '../output/translations.txt', 0)
