#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Wed Jul  8 23:39:22 PDT 2015

"""
Module docstrings.
"""

usage = 'USAGE DESCRIPTION.' 

### Module imports ###
import sys
import os
import argparse # option parsing
import re # regular expression
import codecs
import time
import subprocess
import numpy
### Global variables ###


### Class declarations ###


### Function declarations ###
def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """
  
  parser = argparse.ArgumentParser(description=usage) # add description
  # positional arguments
  parser.add_argument('ref_file', metavar='ref_file', type=str, help='output file') 
  parser.add_argument('trans_file', metavar='trans_file', type=str, help='input file') 

  # optional arguments
  parser.add_argument('-o', '--option', dest='opt', type=int, default=0, help='option (default=0)')
  
  args = parser.parse_args()
  return args

def check_dir(ref_file):
  dir_name = os.path.dirname(ref_file)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def clean_line(line):
  """
  Strip leading and trailing spaces
  """

  line = re.sub('(^\s+|\s$)', '', line);
  return line

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def compute_bleu(bleu_script, lines):
  """
  Compute BLEU score for a set of lines.
  Each line contains: ref\ttrans
  """

  temp_file = 'tmp' + str(int(time.time()))
  temp_ref_file = temp_file + '.ref'
  temp_trans_file = temp_file + '.trans'

  # write to temp file
  f = codecs.open(temp_file, 'w', 'utf-8')
  f.write('\n'.join(lines))
  f.close()
  
  # split to two files
  cmd = 'cut -f 1 %s > %s; cut -f 2 %s > %s;' % (temp_file, temp_ref_file, temp_file, temp_trans_file)
  os.system(cmd)
  
  # compute BLEU
  cmd = ['perl', bleu_script, temp_ref_file]
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
  out = p.communicate(input = open(temp_trans_file).read())[0]
  bleu = out.strip().split()[2].replace(',','')

  os.remove(temp_file)
  os.remove(temp_ref_file)
  os.remove(temp_trans_file)

  return bleu

def score_length(lines, sorted_lens):
  bleus = []
  bleus_cum = []
  lens = []
  total_eval = []
  prev_num_sents = 0
  bleu_script = os.path.dirname(os.path.realpath(__file__)) + '/wmt/multi-bleu.perl'
  sys.stderr.write('# bleu_script = %s\n' % bleu_script)

  group_size = 200
  num_total_sents = len(lines)
  num_sents = 0
  while (1):
    num_sents = num_sents + group_size
    if num_sents>num_total_sents:
      num_sents = num_total_sents

    # individual group score
    bleus.append(compute_bleu(bleu_script, lines[prev_num_sents:num_sents]))
   
    # cumulative score
    bleus_cum.append(compute_bleu(bleu_script, lines[:num_sents]))

    lens.append(numpy.mean(sorted_lens[num_sents-1]))
    total_eval.append(len(lines[prev_num_sents:num_sents]))
    prev_num_sents = num_sents

    if num_sents == num_total_sents:
      break

  print 'bleu\tbleu_cum\tlen\tsize'
  for bleu, bleu_cum, score, num_eval in zip(bleus,bleus_cum,lens,total_eval):
    print bleu + "\t" + bleu_cum + "\t" + repr(score) + "\t" + repr(num_eval)


def process_files(trans_file, ref_file):
  """
  Read data from trans_file, and output to ref_file
  """

  sys.stderr.write('# trans_file = %s, ref_file = %s\n' % (trans_file, ref_file))

  # input
  ref_inf  = codecs.open(ref_file, 'r', 'utf-8')
  ref_lines = ref_inf.readlines()
  ref_inf.close()

  trans_inf = codecs.open(trans_file, 'r', 'utf-8')
  trans_lines = trans_inf.readlines()
  trans_inf.close()

  assert len(trans_lines) == len(ref_lines)
  num_lines = len(trans_lines)
  sys.stderr.write('Num lines = %d\n' % num_lines)
  lines = []
  lens = []
  for ii in xrange(num_lines):
    ref_line = ref_lines[ii].strip()
    trans_line = trans_lines[ii].strip()
    lens.append(len(ref_line.split()))
    lines.append(ref_line + '\t' + trans_line)
   
  # sort
  ids = argsort(lens)
  sorted_lines = []
  sorted_lens = []
  for ii in ids:
    sorted_lines.append(lines[ii])
    sorted_lens.append(lens[ii])

  # score
  score_length(sorted_lines, sorted_lens)

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.trans_file, args.ref_file)
