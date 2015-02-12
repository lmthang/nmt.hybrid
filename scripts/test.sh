#!/bin/bash

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo -e "`basename $0`\tTest LSTM model"
  echo -e "\totherOptions\t\tOther options to test"
  exit
fi

modelFile=$1
beamSize=$2
stackSize=$3
batchSize=$4
outputFile=$5
basicOpt="'$modelFile',$beamSize,$stackSize,$batchSize,'$outputFile'"
if [ $# -eq 6 ]; then
  matlabCommand="test($basicOpt,${6})"
else
  matlabCommand="test($basicOpt)"
fi
echo "$matlabCommand"

echo "mkdir -p $outDir"
mkdir -p $outDir

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# Script dir = $DIR"
echo "cd $DIR/../code"
cd $DIR/../code

echo "$MATLAB -nodesktop -nodisplay -nosplash -r \"$matlabCommand ; exit()\""
$MATLAB -nodesktop -nodisplay -nosplash -r "$matlabCommand ; exit()"

