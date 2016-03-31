#!/bin/sh
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
matlab -nodesktop -nodisplay -nosplash -r "evaluateWordSim('$modelFile', $modelFormat, '$lang');exit;" | tail -n +15


