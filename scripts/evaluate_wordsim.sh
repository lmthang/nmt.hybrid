#!/bin/bash

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo -e "`basename $0`\tTest LSTM model"
  echo -e "\tmodelFile\t\tModel file (.mat)."
  echo -e "\tcharVocab\t\tCharacter vocab."
  echo -e "\toutPrefix\t\tOutput directory."
  echo -e "\tcomputeOpt\t\t 1: char-based, 2: look up embeddings from W_emb_src."
  echo -e "\totherOptions\t\tOther options to test."
  exit
fi

modelFile=$1
charVocab=$2
outPrefix=$3
computeOpt=$4

echo "outPrefix=$outPrefix"
echo "computeOpt=$computeOpt"

outDir=`dirname $outPrefix`
echo "mkdir -p $outDir"
mkdir -p $outDir

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# Script dir = $DIR"

wordsimDir="$DIR/../code/wordsim"

if [[ "$computeOpt" -eq "1" || "$computeOpt" -eq "2" ]]; then # char
  # word to indices
  echo ""
  echo "### word to indices ###"
  wordFile="$outPrefix.opt$computeOpt.words"
  echo "cp $wordsimDir/en_words.txt $wordFile"
  cp $wordsimDir/en_words.txt $wordFile
  wordIndexFile=$outPrefix.indices
  echo "$DIR/../scripts/word_to_char_indices.py $wordFile $charVocab $wordIndexFile"
  $DIR/../scripts/word_to_char_indices.py $wordFile $charVocab $wordIndexFile
  weFile=$outPrefix.opt$computeOpt.We
else
  if [ "$computeOpt" -eq "3" ]; then # embedding lookup
    wordIndexFile=''
    weFile=$outPrefix.opt$computeOpt.wordWe
  fi
fi

# word representations
echo ""
echo "### word representations ###"
if [ $# -eq 5 ]; then
  otherOpts="'$modelFile','$wordIndexFile','$weFile','opt',$computeOpt,$5"
else
  otherOpts="'$modelFile','$wordIndexFile','$weFile','opt',$computeOpt"
fi
matlabCommand="computeSentRepresentations($otherOpts)"
echo "$matlabCommand"

echo "cd $DIR/../code"
cd $DIR/../code
echo "$MATLAB -nodesktop -nodisplay -nosplash -r \"try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()\" > $outPrefix.opt$computeOpt.stderr 2>&1"
$MATLAB -nodesktop -nodisplay -nosplash -r "try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()"  > $outPrefix.opt$computeOpt.stderr 2>&1

# wordsim evaluation
echo ""
echo "### wordsim evaluation ###"
if [[ "$computeOpt" -eq "1" || "$computeOpt" -eq "2" ]]; then # char
  echo "$wordsimDir/code/run_wordSim.sh $outPrefix.opt$computeOpt 3 en"
  $wordsimDir/code/run_wordSim.sh $outPrefix.opt$computeOpt 3 en
else
  if [ "$computeOpt" -eq "3" ]; then # embedding lookup
    echo "$wordsimDir/code/run_wordSim.sh $weFile 2 en"
    $wordsimDir/code/run_wordSim.sh $weFile 2 en
  fi
fi
