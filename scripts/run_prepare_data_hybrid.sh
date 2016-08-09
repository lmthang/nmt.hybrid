#!/bin/bash
# Author: Minh-Thang Luong <luong.m.thang@gmail.com>, created on Fri Nov 14 13:32:54 PST 2014

if [[ $# -lt 5 || $# -gt 5 ]];
then
  echo "`basename $0` trainFile validFile testFile vocabFile outDir" 
  exit 1
fi

trainFile=$1
validFile=$2
testFile=$3
vocabFile=$4
outDir=$5
  
freq=0


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
vocabStr=""

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

