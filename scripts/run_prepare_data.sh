#!/bin/bash
# Author: Minh-Thang Luong <luong.m.thang@gmail.com>, created on Fri Nov 14 13:32:54 PST 2014

if [[ $# -lt 5 || $# -gt 7 ]];
then
  echo "`basename $0` trainFile validFile testFile vocabSize outDir [vocabFile] [freq]" 
  exit 1
fi

trainFile=$1
validFile=$2
testFile=$3
vocabSize=$4
outDir=$5
  

vocabFile=""
if [ $# -ge 6 ]; then
  vocabFile=$6
fi
freq=0
if [ $# -ge 7 ]; then
  freq=$7
fi



SCRIPT_DIR=$(dirname $0)

function execute_check {
  file=$1 
  cmd=$2
  
  if [[ -f $file || -d $file ]];
  then
    echo ""
    echo "! File/directory $file exists. Skip."
  else
    echo ""
    echo "# Executing: $cmd"
    eval $cmd
  fi
}

# check outDir exists
echo "# outDir $outDir"
execute_check $outDir "mkdir -p $outDir"

# vocab
if [ "$vocabFile" = "" ]; then
  basePrefix=`basename $trainFile`
    vocabFile="$outDir/$basePrefix.vocab"

  if [ $freq -eq 0 ]; then
    vocabStr="--size $vocabSize"
  else
    vocabStr="--freq $freq"
  fi
  echo "vocabStr=$vocabStr"
fi

# train
trainName=`basename $trainFile`
outFile="$outDir/$trainName"
execute_check "$outFile" "$SCRIPT_DIR/prepare_data.py --vocab_file $vocabFile $vocabStr $trainFile $outFile"
execute_check "$outFile.reversed" "$SCRIPT_DIR/reverse.py $outFile $outFile.reversed"

# valid
validName=`basename $validFile`
outFile="$outDir/$validName"
execute_check "$outFile" "$SCRIPT_DIR/prepare_data.py --vocab_file $vocabFile $vocabStr $validFile $outFile"
execute_check "$outFile.reversed" "$SCRIPT_DIR/reverse.py $outFile $outFile.reversed"

# test
testName=`basename $testFile`
outFile="$outDir/$testName"
execute_check "$outFile" "$SCRIPT_DIR/prepare_data.py --vocab_file $vocabFile $vocabStr $testFile $outFile"
execute_check "$outFile.reversed" "$SCRIPT_DIR/reverse.py $outFile $outFile.reversed"

