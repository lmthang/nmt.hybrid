#!/bin/sh

trans_dir=$1
~/nmt.matlab/scripts/post_process.py ~/deepmt/data/iwslt15/en-de/IWSLT15.TED.tst2013.en-de.tok.en $trans_dir/translations.txt $trans_dir/translations.txt.align ~/deepmt/data/deen/wmt/wmt-combined.deen.tok.filtered.f5.new.en-de.dict ~/deepmt/data/iwslt15/en-de/IWSLT15.TED.tst2013.en-de.tok.unsplit.de '' -l german -s ~/deepmt/data/iwslt15/en-de/IWSLT15.TED.tst2013.en-de.en.xml -t ~/deepmt/data/iwslt15/en-de/IWSLT15.TED.tst2013.en-de.de.xml
