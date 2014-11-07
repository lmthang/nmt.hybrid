Code to train Long-Short Term Memory (LSTM) models

Thang Luong <lmthang@stanford.edu>, 2014

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

(b) Train an LSTM
run.sh  Train RNN models
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

The script run.sh will call the main LSTM training code at code/trainLSTM.m

/*********************/
/** Sample commands **/
/*********************/
(a) Prepare the data
mkdir output
./scripts/prepare_data.py --size 1000 ./data/train.10k.en ./output/train.10k.id.en 
./scripts/prepare_data.py --vocab_file ./data/train.10k.en.vocab.1000 ./data/valid.3k.en ./output/valid.3k.id.en 
./scripts/prepare_data.py --vocab_file ./data/train.10k.en.vocab.1000 ./data/test.3k.en ./output/test.3k.id.en 

./scripts/prepare_data.py --size 1000 ./data/train.10k.de ./output/train.10k.id.de 
./scripts/prepare_data.py --vocab_file ./data/train.10k.de.vocab.1000 ./data/valid.3k.de ./output/valid.3k.id.de 
./scripts/prepare_data.py --vocab_file ./data/train.10k.de.vocab.1000 ./data/test.3k.de ./output/test.3k.id.de 

(b) Train an LSTM model
export MATLAB=matlab
./scripts/run.sh ../output/train.10k.id ../output/valid.3k.id ../output/test.3k.id de en ../data/train.10k.en.vocab.1000 ../data/train.10k.de.vocab.1000 ../output 0 100 0.1 5 0.1 128 10 1

To run directly in Matlab, cd into code/ directory and run:
trainLSTM('../output/train.10k.id', '../output/valid.3k.id', '../output/test.3k.id', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 0, 'logFreq', 1)

(c) Grad check
trainLSTM('', '', '', '', '', '', '', '', 0, 'isGradCheck', 1)
Note: the grad check works on CPU. For GPU, you will need to change this line "dataType = 'single';" into 'double' and remove the single conversion in the functions clipForward() and clipBackward().

(d) Profiling
trainLSTM('../output/train.10k.id', '../output/valid.3k.id', '../output/test.3k.id', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 0, 'logFreq', 1, 'isProfile', 1)

(e) Compare results
The output of (c) on CPU should match content of the file data/gradCheck.output.
The output of (d) on CPU should match the content of the file data/sample.output.

(f) Run on full data
trainLSTM('../data/merged_training.40k.id', '../data/tiny_tune.40k.id', '../data/p1r6_dev.40k.id', 'zh', 'en', '../data/merged_training.zh.vocab.40000', '../data/merged_training.en.vocab.40000', '../output', 0, 'logFreq', 1)

