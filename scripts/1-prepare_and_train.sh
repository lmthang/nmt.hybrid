#!/bin/bash

if [[ $# -lt 9 || $# -gt 10 ]]; then
  echo -e "`basename $0`\tTrain LSTM models"
  echo -e "\ttrainPrefix\t\texpect train files trainPrefix.(srcLang|tgtLang)"
  echo -e "\tvalidPrefix\t\texpect valid files validPrefix.(srcLang|tgtLang)"
  echo -e "\ttestPrefix\t\texpect test files testPrefix.(srcLang|tgtLang)"
  echo -e "\tsrcLang\t\t\tSource languague"
  echo -e "\ttgtLang\t\t\tTarget languague"
  echo -e "\twordVocabSize\t\tWord vocab size."
  echo -e "\tcharVocabSize\t\tCharacter vocab size. If 0, run word-based models."
  echo -e "\toutDataDir\t\tOutput data directory where we save preprocessed data"
  echo -e "\toutModelDir\t\tOutput model directory that we save during training"
  echo -e "\toptions\t\t\tOptions to trainLSTM"
  exit
fi

trainPrefix=$1
validPrefix=$2
testPrefix=$3
srcLang=$4
tgtLang=$5
wordVocabSize=$6
charVocabSize=$7
outDataDir=$8
outModelDir=$9
if [ $# -eq 10 ]; then
  options=",${10}"
else
  options=""
fi

function execute_check {
  file=$1 
  cmd=$2
  
  if [[ -f $file || -d $file ]];
  then
    echo ""
    echo "! File/directory $file exists. Skip."
  else
    echo ""
    echo "## Executing: $cmd"
    eval $cmd
  fi
}

# check MATLAB
if [ "$MATLAB" = "" ]; then
  echo "Need to set the environment variable MATLAB!"
  exit
fi

# check outDataDir exists
echo "# outDataDir $outDataDir"
execute_check $outDataDir "mkdir -p $outDataDir"

# go to the code directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# Script dir = $SCRIPT_DIR"

# First, extract vocabulary files:
# extract_vocab.py <input_text_file> <output_vocab_file>
srcVocabFile="$outDataDir/$srcLang.vocab"
execute_check $srcVocabFile "$SCRIPT_DIR/extract_vocab.py $trainPrefix.$srcLang $srcVocabFile" 
tgtVocabFile="$outDataDir/$tgtLang.vocab"
execute_check $tgtVocabFile "$SCRIPT_DIR/extract_vocab.py $trainPrefix.$tgtLang $tgtVocabFile" 

# Next, we convert train/valid/test files in text format into integer format
# that can be handled efficiently in Matlab:
idDir="$outDataDir/id"
trainPrefix_id=`basename $trainPrefix`
trainPrefix_id="$idDir/$trainPrefix_id"
validPrefix_id=`basename $validPrefix`
validPrefix_id="$idDir/$validPrefix_id"
testPrefix_id=`basename $testPrefix`
testPrefix_id="$idDir/$testPrefix_id"
if [ $charVocabSize -gt 0 ]; then # hybrid models
  # build_char_dict.py <vocab_file> <output_name_prefix> <char_vocab_size> 
  srcCharPrefix="$outDataDir/$srcLang.$charVocabSize"
  execute_check "$srcCharPrefix.char.vocab" "$SCRIPT_DIR/build_char_dict.py $srcVocabFile $srcCharPrefix $charVocabSize"
  tgtCharPrefix="$outDataDir/$tgtLang.$charVocabSize"
  execute_check "$tgtCharPrefix.char.vocab" "$SCRIPT_DIR/build_char_dict.py $tgtVocabFile $tgtCharPrefix $charVocabSize"

  # now update vocab files with new filtered vocabs (that are covered by our char vocabs)
  srcVocabFile="$srcCharPrefix.vocab"
  tgtVocabFile="$tgtCharPrefix.vocab"

  # run_prepare_data_hybrid.sh <trainFile> <valid_file> <test_file> <vocab_file> <out_dir>
  execute_check "$trainPrefix_id.$srcLang" "$SCRIPT_DIR/run_prepare_data_hybrid.sh $trainPrefix.$srcLang $validPrefix.$srcLang $testPrefix.$srcLang $srcVocabFile $idDir"
  execute_check "$trainPrefix_id.$tgtLang" "$SCRIPT_DIR/run_prepare_data_hybrid.sh $trainPrefix.$tgtLang $validPrefix.$tgtLang $testPrefix.$tgtLang $tgtVocabFile $idDir"
else
  # run_prepare_data.sh <trainFile> <valid_file> <test_file> <vocab_file> <out_dir>
  execute_check "$trainPrefix_id.$srcLang" "$SCRIPT_DIR/run_prepare_data.sh $trainPrefix.$srcLang $validPrefix.$srcLang $testPrefix.$srcLang $wordVocabSize $idDir"
  execute_check "$trainPrefix_id.$tgtLang" "$SCRIPT_DIR/run_prepare_data.sh $trainPrefix.$tgtLang $validPrefix.$tgtLang $testPrefix.$tgtLang $wordVocabSize $idDir"
  srcVocabFile="$trainPrefix_id.$srcLang.vocab.$wordVocabSize"
  tgtVocabFile="$trainPrefix_id.$tgtLang.vocab.$wordVocabSize"
fi

# Executing matlab code
echo ""
echo "## Run MATLAB ##"
MATLAB_DIR="$SCRIPT_DIR/../code"
#echo "cd $SCRIPT_DIR/../code"
# cd $SCRIPT_DIR/../code

# hybrid model options
hybridOptions=""
if [ $charVocabSize -gt 0 ]; then
  hybridOptions=",'charOpt',3,'charFeedOpt',1,'srcCharShortList',$wordVocabSize,'tgtCharShortList',$wordVocabSize,'srcCharPrefix','$srcCharPrefix','tgtCharPrefix', '$tgtCharPrefix'"
fi
matlabCommand="trainLSTM('$trainPrefix_id','$validPrefix_id','$testPrefix_id','$srcLang','$tgtLang','$srcVocabFile','$tgtVocabFile','$outModelDir'${hybridOptions}${options})"

execute_check "" "$MATLAB -nodesktop -nodisplay -nosplash -r \"try; addpath(genpath('$MATLAB_DIR')); $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()\""

