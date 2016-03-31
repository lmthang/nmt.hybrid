#!/bin/sh

if [[ ! $# -eq 2 && ! $# -eq 2 ]]
then
    echo "`basename $0` wordVectorFile outPrefix" 
    exit
fi

wordVectorFile=$1
outPrefix=$2
outDir=`dirname $outPrefix`

# check outDir exists
#echo "# outDir $outDir"
if [ ! -d $outDir ] 
then
#  echo "# Directory exists $outDir"
#else
  mkdir -p $outDir
fi


inFile=$1
numWords=`tail -1 $inFile | wc -w`
let numDimensions=numWords-1
#echo "numDimensions=$numDimensions"

numWordsFirstLine=`head -1 $inFile | wc -w`
if [ $numWordsFirstLine -ne $numWords ]; then # remove the first line (needed for output from word2vec where the first line contains the number words and the dimension)
  #echo "tail -n +2 $inFile > $inFile.tmp"
  tail -n +2 $inFile > $inFile.tmp
  inFile=$inFile.tmp
fi

cut -d " " -f 1 $inFile > $outPrefix.words
cut -d " " -f 2- $inFile > $outPrefix.We

if [ $numWordsFirstLine -ne $numWords ]; then
  rm -rf $inFile
fi

