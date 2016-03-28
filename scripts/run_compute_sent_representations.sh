#!/bin/bash

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo -e "`basename $0`\tTest LSTM model"
  echo -e "\tmodelFile\t\tModel file (.mat)."
  echo -e "\tinFile\t\tInput file (integers)."
  echo -e "\toutputFile\t\toutput file for the final translations."
  echo -e "\totherOptions\t\tOther options to test."
  exit
fi

modelFile=$1
inFile=$2
outputFile=$3
basicOpt="'$modelFile','$inFile','$outputFile'"
if [ $# -eq 4 ]; then
  matlabCommand="computeSentRepresentations($basicOpt,${4})"
else
  matlabCommand="computeSentRepresentations($basicOpt)"
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

