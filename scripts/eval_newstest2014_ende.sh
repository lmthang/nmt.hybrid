#!/bin/sh

trans_dir=$1
~/lstm15/scripts/post_process.py ~/deepmt/data/deen/wmt/newstest2014.deen.tok.en $trans_dir/translations.txt $trans_dir/translations.txt.align ~/deepmt/data/deen/wmt/wmt-combined.deen.tok.filtered.f5.new.en-de.dict ~/deepmt/data/deen/wmt/newstest2014.deen.tok.de '' -l de -s ~/deepmt/data/deen/wmt/newstest2014-deen-src.en.sgm -t  ~/deepmt/data/deen/wmt/newstest2014-deen-ref.de.sgm 
