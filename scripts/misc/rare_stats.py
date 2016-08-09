#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Tue Nov 24 18:58:35 PST 2015

"""
Build mapping from words into sequences of characters.
"""

usage = 'Build mapping from words into sequences of characters.' 

### Module imports ###
import sys
import os
import argparse # option parsing
import re # regular expression
import codecs

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
  parser.add_argument('short_list_sizes', metavar='short_list_sizes', type=str, help='list of short-list sizes, comma-separated') 

  # optional arguments
  parser.add_argument('-o', '--option', dest='opt', type=int, default=0, help='option (default=0)')
  
  args = parser.parse_args()
  return args

def clean_line(line):
  """
  Strip leading and trailing spaces
  """

  line = re.sub('(^\s+|\s$)', '', line);
  return line

def process_files(in_file, short_list_sizes_str):
  """
  Read data from in_file
  """

  sys.stderr.write('# in_file = %s, short_list_sizes = %s\n' % (in_file, short_list_sizes_str))

  # input
  sys.stderr.write('# Input from %s.\n' % (in_file))
  inf = codecs.open(in_file, 'r', 'utf-8')

 
  short_list_sizes = [int(token) for token in short_list_sizes_str.split(',')]
  rare_counts = []
  rare_distinct_counts = []
  num_sizes = len(short_list_sizes)
  for _ in xrange(num_sizes):
    rare_counts.append(0)
    rare_distinct_counts.append(0)

  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
  num_words = 0
  for line in inf:
    # init
    rare_words = []
    sent_rare_counts = []
    for _ in xrange(num_sizes):
      rare_words.append({})
      sent_rare_counts.append(0)
    
    # count rare words
    tokens = line.strip().split()
    num_words += len(tokens)
    for token in tokens:
      index = int(token)
      for i in xrange(num_sizes):
        if index >= short_list_sizes[i]:
          sent_rare_counts[i] += 1
          rare_words[i][index] = 1
    
    # update total
    for i in xrange(num_sizes):
      rare_counts[i] += sent_rare_counts[i]
      rare_distinct_counts[i] += len(rare_words[i])

    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%dK) ' % (line_id/1000))
  inf.close()
  line_id = float(line_id)
  sys.stderr.write('Done! Num sents %d, num avg words %.2f\n' % (line_id, num_words/line_id))

  for i in xrange(num_sizes):
    sys.stderr.write('# short list %d: num avg rare words %.2f, num avg distinct rare words %.2f\n' % (short_list_sizes[i], rare_counts[i] / line_id, rare_distinct_counts[i] / line_id))

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.short_list_sizes)

