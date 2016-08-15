#!/bin/bash

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo -e "`basename $0`\tTest LSTM model"
  echo -e "\tmodelFile\t\tModel file (.mat)."
  echo -e "\tbeamSize\t\tbeam size, e.g., 12."
  echo -e "\tstackSize\t\tnumber of top translations to keep, e.g., 100."
  echo -e "\tbatchSize\t\tnumber of sentences to be decoded each time, e.g., 10."
  echo -e "\toutputFile\t\toutput file for the final translations."
  echo -e "\totherOptions\t\tOther options to test."
  exit
fi

modelFile=$1
beamSize=$2
stackSize=$3
batchSize=$4
outputFile=$5
basicOpt="'$modelFile',$beamSize,$stackSize,$batchSize,'$outputFile'"
if [ $# -eq 6 ]; then
  matlabCommand="testLSTM($basicOpt,${6})"
else
  matlabCommand="testLSTM($basicOpt)"
fi
echo "$matlabCommand"

outDir=`dirname $outputFile`
echo "mkdir -p $outDir"
mkdir -p $outDir

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# Script dir = $DIR"
echo "cd $DIR/../code"
cd $DIR/../code

echo "$MATLAB -nodesktop -nodisplay -nosplash -r \"try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()\""
$MATLAB -nodesktop -nodisplay -nosplash -r "try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()"

