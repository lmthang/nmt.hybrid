#!/bin/sh
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Thu May 28 23:36:15 PDT 2015

if [[ ! $# -eq 1 && ! $# -eq 1 ]]
then
    echo "`basename $0` outDir [verbose]" 
    exit
fi

outDir=$1
VERBOSE=1
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

echo "# outDir $outDir"

cd $outDir
mv train/wmt.ende.* .
mv test/newstest* .
mv valid/newstest* .

mkdir id.50000
mv wmt.ende.id.de id.50000/wmt.ende.de
mv wmt.ende.id.en id.50000/wmt.ende.en
mv newstest2013.deen.tok.id.de id.50000/newstest2013.deen.tok.de
mv newstest2013.deen.tok.id.en id.50000/newstest2013.deen.tok.en
mv newstest2014.deen.tok.id.de id.50000/newstest2014.deen.tok.de
mv newstest2014.deen.tok.id.en id.50000/newstest2014.deen.tok.en
ln -s wmt.ende.vocab.de wmt.ende.de.vocab.50000
ln -s wmt.ende.vocab.en wmt.ende.en.vocab.50000
$SCRIPT_DIR/reverse.py $outDir/id.50000/wmt.ende.en $outDir/id.50000/wmt.ende.reversed.en
$SCRIPT_DIR/reverse.py $outDir/id.50000/newstest2013.deen.tok.en $outDir/id.50000/newstest2013.deen.tok.reversed.en
$SCRIPT_DIR/reverse.py $outDir/id.50000/newstest2014.deen.tok.en $outDir/id.50000/newstest2014.deen.tok.reversed.en
