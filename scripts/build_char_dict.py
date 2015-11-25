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
  parser.add_argument('out_prefix', metavar='out_prefix', type=str, help='output file') 

  # optional arguments
  parser.add_argument('-o', '--option', dest='opt', type=int, default=0, help='option (default=0)')
  
  args = parser.parse_args()
  return args

def check_dir(out_prefix):
  dir_name = os.path.dirname(out_prefix)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def clean_line(line):
  """
  Strip leading and trailing spaces
  """

  line = re.sub('(^\s+|\s$)', '', line);
  return line

def process_files(in_file, out_prefix):
  """
  Read data from in_file, and output to out_prefix
  """

  sys.stderr.write('# in_file = %s, out_prefix = %s\n' % (in_file, out_prefix))
  # input
  sys.stderr.write('# Input from %s.\n' % (in_file))
  inf = codecs.open(in_file, 'r', 'utf-8')

  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
  vocabs = []
  char_dict = {}
  char_map = {}
  num_chars = 0
  for line in inf:
    word = clean_line(line)
    vocabs.append(word)
    for char in word:
      if char not in char_dict:
        char_dict[char] = 0
        char_map[char] = num_chars
        num_chars += 1
      char_dict[char] += 1

    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)
  sys.stderr.write('Done! Num words = %d, num chars = %d\n' % (line_id, num_chars))
  inf.close()
  
  # output
  check_dir(out_prefix)
  char_out_file = out_prefix + '.char'
  char_map_out_file = out_prefix + '.char.map'
  sys.stderr.write('Output to %s, %s\n' % (char_out_file, char_map_out_file))
  char_ouf = codecs.open(char_out_file, 'w', 'utf-8')
  char_map_ouf = codecs.open(char_map_out_file, 'w', 'utf-8')

  for word in vocabs:
    char_map_ouf.write('%s\n' % ' '.join([str(char_map[char]) for char in word]))
  
  for (k, v) in sorted(char_dict.items(), key=lambda x: x[1], reverse=True):
    char_ouf.write('%s %d\n' % (k, v))
    
  char_ouf.close()
  char_map_ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.out_prefix)

