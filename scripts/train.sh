#!/bin/bash

if [[ $# -lt 15 || $# -gt 16 ]]; then
  echo -e "`basename $0`\tTrain LSTM models"
  echo -e "\ttrainPrefix\t\texpect train files trainPrefix.(srcLang|tgtLang)"
  echo -e "\tvalidPrefix\t\texpect valid files validPrefix.(srcLang|tgtLang)"
  echo -e "\ttestPrefix\t\texpect test files testPrefix.(srcLang|tgtLang)"
  echo -e "\tsrcLang\t\t\tSource languague"
  echo -e "\ttgtLang\t\t\tTarget languague"
  echo -e "\tsrcVocabFile\t\t\tSource vocab file"
  echo -e "\ttgtVocabFile\t\t\tTarget vocab file"
  echo -e "\toutDir\t\t\tOutput directory"
  echo -e "\tlstmSize\t\tDimension of source word vectors"
  echo -e "\tlearningRate\t\tLearning rate"
  echo -e "\tmaxGradNorm\t\tMax grad norm"
  echo -e "\tinitRange\t\tNumber of features for source binary tree traversal"
  echo -e "\tbatchSize\t\tNumber of sentences per minibatch, larger gives faster training time but worse results"
  echo -e "\tnumEpoches\t\tNumber of training epochs"
  echo -e "\tlogFreq\t\t\tCompute validation perplexities after [logFreq] dots printed"
  echo -e "\totherOptions\t\tOther options to trainLSTM"
  exit
fi

trainPrefix=$1
validPrefix=$2
testPrefix=$3
srcLang=$4
tgtLang=$5
srcVocabFile=$6
tgtVocabFile=$7
outDir=$8
lstmSize=$9
learningRate=${10}
maxGradNorm=${11}
initRange=${12}
batchSize=${13}
numEpoches=${14}
logFreq=${15}
basicOpt="'$trainPrefix','$validPrefix','$testPrefix','$srcLang','$tgtLang','$srcVocabFile','$tgtVocabFile','$outDir','lstmSize',$lstmSize,'maxGradNorm',$maxGradNorm,'learningRate',$learningRate,'initRange',$initRange,'batchSize',$batchSize,'numEpoches',$numEpoches,'logFreq',$logFreq"
if [ $# -eq 16 ]; then
  matlabCommand="trainLSTM($basicOpt,${16})"
else
  matlabCommand="trainLSTM($basicOpt)"
fi
echo "$matlabCommand"

echo "mkdir -p $outDir"
mkdir -p $outDir

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# Script dir = $DIR"
echo "cd $DIR/../code"
cd $DIR/../code

#MATLAB="matlab"
echo "$MATLAB -nodesktop -nodisplay -nosplash -r \"try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()\""
$MATLAB -nodesktop -nodisplay -nosplash -r "try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()"

