#!/bin/bash

if [[ $# -lt 16 || $# -gt 17 ]]; then
  echo -e "`basename $0`\tTrain LSTM models"
  echo -e "\ttrainPrefix\t\texpect train files trainPrefix.(srcLang|tgtLang)"
  echo -e "\tvalidPrefix\t\texpect valid files validPrefix.(srcLang|tgtLang)"
  echo -e "\ttestPrefix\t\texpect test files testPrefix.(srcLang|tgtLang)"
  echo -e "\tsrcLang\t\t\tSource languague"
  echo -e "\ttgtLang\t\t\tTarget languague"
  echo -e "\tsrcVocabFile\t\t\tSource vocab file"
  echo -e "\ttgtVocabFile\t\t\tTarget vocab file"
  echo -e "\toutDir\t\t\tOutput directory"
  echo -e "\tbaseIndex\t\t\tBase index"
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

#MATLAB=/afs/cs/software/bin/matlab_r2013b
trainPrefix=$1
validPrefix=$2
testPrefix=$3
srcLang=$4
tgtLang=$5
srcVocabFile=$6
tgtVocabFile=$7
outDir=$8
baseIndex=$9
lstmSize=${10}
learningRate=${11}
maxGradNorm=${12}
initRange=${13}
batchSize=${14}
numEpoches=${15}
logFreq=${16}
basicOpt="'$trainPrefix','$validPrefix','$testPrefix','$srcLang','$tgtLang','$srcVocabFile','$tgtVocabFile','$outDir',$baseIndex,'lstmSize',$lstmSize,'maxGradNorm',$maxGradNorm,'learningRate',$learningRate,'initRange',$initRange,'batchSize',$batchSize,'numEpoches',$numEpoches,'logFreq',$logFreq"
if [ $# -eq 17 ]; then
  matlabCommand="trainLSTM($basicOpt,${17})"
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

$MATLAB -nodesktop -nodisplay -nosplash -r "$matlabCommand ; exit()"  # 

