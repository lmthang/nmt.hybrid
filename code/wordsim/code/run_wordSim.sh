#!/bin/bash
# Author: Minh-Thang Luong <luong.m.thang@gmail.com>, created on Thu Feb 21 16:32:43 PST 2013

if [[ ! $# -eq 3 && ! $# -eq 3 ]]
then
    echo "`basename $0` modelFile modelFormat lang"
    echo "lang: en or de"
    echo "modelFormat : 0 -- Matlab file,"
    echo "      1 -- text file with a header line <numWords> <embDim>.  Subsequent lines has <word> <values>."
    echo "      2 -- text file with each line has <word> <values>"
    echo "      3 -- assume that there are two files modelFile.We, modelFile.words"
    exit
fi

modelFile=$1
modelFormat=$2
lang=$3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

if [ "$modelFormat" -eq 1 ]; then # split We, words
  #echo "./splitWordVectorFile.sh $modelFile $modelFile"
  ./splitWordVectorFile.sh $modelFile $modelFile
  modelFormat=3
fi

matlabCommand="evaluateWordSim('$modelFile', $modelFormat, '$lang')"
echo "$MATLAB -nodesktop -nodisplay -nosplash -r \"try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()\""
$MATLAB -nodesktop -nodisplay -nosplash -r "try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()"
