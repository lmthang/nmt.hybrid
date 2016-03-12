#!/bin/bash

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo -e "`basename $0`\tGenerate new model file w.r.t new vocab files"
  echo -e "\tmodelFile\t\tModel file (.mat)."
  echo -e "\tsrcVocabFile\tnew src vocab file."
  echo -e "\ttgtVocabFile\tnew tgt vocab file."
  echo -e "\tsrcCharPrefix\tnew src char prefix."
  echo -e "\ttgtCharPrefix\tnew tgt char prefix."
  echo -e "\toutputFile\t\toutput new model file"
  exit
fi

modelFile=$1
srcVocabFile=$2
tgtVocabFile=$3
srcCharPrefix=$4
tgtCharPrefix=$5
outputFile=$6
basicOpt="'$modelFile','$srcVocabFile','$tgtVocabFile','srcCharPrefix','tgtCharPrefix','$outputFile'"
if [ $# -eq 7 ]; then
  matlabCommand="transferModel($basicOpt,${7})"
else
  matlabCommand="transferModel($basicOpt)"
fi
echo "$matlabCommand"

outDir=`dirname $outputFile`
echo "mkdir -p $outDir"
mkdir -p $outDir

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# Script dir = $DIR"
echo "cd $DIR/../code/misc"
cd $DIR/../code/misc

echo "$MATLAB -nodesktop -nodisplay -nosplash -r \"try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()\""
$MATLAB -nodesktop -nodisplay -nosplash -r "try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()"

