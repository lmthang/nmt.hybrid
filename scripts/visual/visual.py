#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Wed Jun  3 01:22:18 MDT 2015

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
from tsne import bh_sne
import numpy as np
#sys.path.append(os.environ['HOME'] + '/lib/') # add our own libraries

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
  parser.add_argument('in_file', metavar='in_file', type=str, help='input file') 
  parser.add_argument('out_file', metavar='out_file', type=str, help='output file') 

  # optional arguments
  parser.add_argument('-o', '--option', dest='opt', type=int, default=0, help='option (default=0)')
  
  args = parser.parse_args()
  return args

def check_dir(out_file):
  dir_name = os.path.dirname(out_file)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def clean_line(line):
  """
  Strip leading and trailing spaces
  """

  line = re.sub('(^\s+|\s$)', '', line);
  return line

def process_files(in_file, out_file):
  """
  Read data from in_file, and output to out_file
  """

  sys.stderr.write('# in_file = %s, out_file = %s\n' % (in_file, out_file))
  # input
  sys.stderr.write('# Input from %s.\n' % (in_file))
  inf = codecs.open(in_file, 'r', 'utf-8')

  # output
  sys.stderr.write('Output to %s\n' % out_file)
  check_dir(out_file)
  ouf = codecs.open(out_file, 'w', 'utf-8')

  line_id = 0
  words = []
  embs = []
  num_dim = -1
  all_lines = inf.readlines()
  num_words = len(all_lines)
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
  sys.stderr.write('# num words = %d\n' % (num_words))
  for line in all_lines:
    line = clean_line(line)
    tokens = re.split('\s+', line)
    word = tokens[0]
    if line_id==0:
      num_dim = len(tokens)-1
      sys.stderr.write('# num dims = %d\n' % (num_dim))
      X = np.zeros((num_words, num_dim))
    emb = np.array(tokens[1:], dtype='|S4')
    emb = emb.astype(np.float)
    X[line_id, :] = emb

    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)

  sys.stderr.write('Done! Num lines = %d\n' % line_id)

  X_2d = bh_sne(X)
  for ii in xrange(num_words):
    ouf.write('%f %f\n' % (X_2d[ii, 0], X_2d[ii, 1]))
  inf.close()
  ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.out_file)
