#!/bin/bash
# Author: Minh-Thang Luong <luong.m.thang@gmail.com>, created on Fri Nov 14 13:32:54 PST 2014

if [[ ! $# -eq 5 && ! $# -eq 5 ]];
then
    echo "`basename $0` trainFile validFile testFile vocabSize outDir" 
    exit
fi

trainFile=$1
validFile=$2
testFile=$3
vocabSize=$4
outDir=$5
sizeStr="--size $vocabSize"
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
basePrefix=`basename $trainFile`
vocabFile="$outDir/$basePrefix.vocab.$vocabSize"

# train
trainName=`basename $trainFile`
outFile="$outDir/$trainName"
execute_check "$outFile" "$SCRIPT_DIR/prepare_data.py --vocab_file $vocabFile $sizeStr $trainFile $outFile"

# valid
validName=`basename $validFile`
outFile="$outDir/$validName"
execute_check "$outFile" "$SCRIPT_DIR/prepare_data.py --vocab_file $vocabFile $sizeStr $validFile $outFile"

# test
testName=`basename $testFile`
outFile="$outDir/$testName"
execute_check "$outFile" "$SCRIPT_DIR/prepare_data.py --vocab_file $vocabFile $sizeStr $testFile $outFile"
