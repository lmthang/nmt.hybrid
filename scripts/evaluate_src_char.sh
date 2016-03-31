#!/bin/bash

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo -e "`basename $0`\tTest LSTM model"
  echo -e "\tmodelFile\t\tModel file (.mat)."
  echo -e "\tcharVocab\t\tCharacter vocab."
  echo -e "\toutPrefix\t\tOutput directory."
  echo -e "\totherOptions\t\tOther options to test."
  exit
fi

modelFile=$1
charVocab=$2
outPrefix=$3

outDir=`dirname $outPrefix`
echo "mkdir -p $outDir"
mkdir -p $outDir

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# Script dir = $DIR"

# word to indices
echo ""
echo "### word to indices ###"
wordsimDir="$DIR/../code/wordsim"
wordFile="$outPrefix.words"
echo "cp $wordsimDir/en_words.txt $wordFile"
cp $wordsimDir/en_words.txt $wordFile
wordIndexFile=$outPrefix.indices
echo "$DIR/../scripts/word_to_char_indices.py $wordFile $charVocab $wordIndexFile"
$DIR/../scripts/word_to_char_indices.py $wordFile $charVocab $wordIndexFile

# word representations
echo ""
echo "### word representations ###"
weFile=$outPrefix.We
if [ $# -eq 4 ]; then
  opts="'$modelFile','$wordIndexFile','$weFile',$4"
else
  opts="'$modelFile','$wordIndexFile','$weFile'"
fi
matlabCommand="computeSentRepresentations($opts)"
echo "$matlabCommand"

echo "cd $DIR/../code"
cd $DIR/../code
echo "$MATLAB -nodesktop -nodisplay -nosplash -r \"try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()\" > $outPrefix.stderr 2>&1"
$MATLAB -nodesktop -nodisplay -nosplash -r "try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()"  > $outPrefix.stderr 2>&1

# wordsim evaluation
echo ""
echo "### wordsim evaluation ###"
echo "$wordsimDir/code/run_wordSim.sh $outPrefix 3 en"
$wordsimDir/code/run_wordSim.sh $outPrefix 3 en
