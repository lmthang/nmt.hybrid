#!/bin/sh
# Author: Minh-Thang Luong <luong.m.thang@gmail.com>, created on Fri Nov 14 13:32:54 PST 2014

if [[ ! $# -eq 6 && ! $# -eq 5 ]]
then
    echo "`basename $0` trainFile validFile testFile vocabSize outDir [verbose]" 
    exit
fi

trainFile=$1
validFile=$2
testFile=$3
vocabSize=$4
outDir=$5
VERBOSE=1
if [ $# -eq 6 ]; then
  VERBOSE=$6
fi
SCRIPT_DIR=$(dirname $0)

function execute_check {
  file=$1 
  cmd=$2
  
  if [[ -f $file || -d $file ]]; then
    echo ""
    echo "! File/directory $file exists. Skip."
  else
    echo ""
    if [ $VERBOSE -eq 1 ]; then
      echo "# Executing: $cmd"
    fi
    
    eval $cmd
  fi
}

# check outDir exists
echo "# outDir $outDir"
execute_check $outDir "mkdir -p $outDir"

# train
trainName=`basename $trainFile`
outFile="$outDir/$trainName"
execute_check "$outFile" "$SCRIPT_DIR/prepare_data.py --size $vocabSize $trainFile $outFile"

# valid
validName=`basename $validFile`
outFile="$outDir/$validName"
execute_check "$outFile" "$SCRIPT_DIR/prepare_data.py --vocab_file $trainFile.vocab.$vocabSize --size $vocabSize $trainFile $outFile"

# test
testName=`basename $testFile`
outFile="$outDir/$testName"
execute_check "$outFile" "$SCRIPT_DIR/prepare_data.py --vocab_file $trainFile.vocab.$vocabSize --size $vocabSize $trainFile $outFile"
