#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Tue Jun  2 23:55:16 PDT 2015

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
  parser.add_argument('word_list_file', metavar='word_list_file', type=str, help='list of words') 
  parser.add_argument('out_file', metavar='out_file', type=str, help='output file') 

  # optional arguments
  parser.add_argument('-o', '--option', dest='opt', type=int, default=0,
      help='0: in_file contains both words and embeddings with a header line, 1: expect files in_file.We, in_file.words (default=0)')
  
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

def process_files(in_file, word_list_file, out_file, opt):
  """
  Read data from in_file, and output to out_file
  """

  sys.stderr.write('# in_file = %s, out_file = %s, opt = %d\n' % (in_file,
    out_file, opt))
  if opt == 0:
    inf = codecs.open(in_file, 'r', 'utf-8')
    inf.readline() # skip header line
  else:
    inf = codecs.open(in_file + '.words', 'r', 'utf-8')
    inf_We = open(in_file + '.We', 'r')
 
  ouf = codecs.open(out_file, 'w', 'utf-8')
  check_dir(out_file)


  # load word list
  words = {}
  word_inf = codecs.open(word_list_file, 'r', 'utf-8')
  for line in word_inf:
    words[clean_line(line)] = 1
  word_inf.close()

  # output

  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
  all_words = []
  for line in inf:
    line = clean_line(line)
    if opt == 0:
      tokens = re.split('\s+', line)
      all_words.append(tokens[0])
      if tokens[0] in words:
        ouf.write('%s\n' % line)
    else:
      all_words.append(line)
      We_line = inf_We.readline()
      if line in words:
        ouf.write('%s %s' % (line, We_line))
      
    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)
  sys.stderr.write('Done! Num lines = %d\n' % line_id)

  for word in words:
    if word not in all_words:
      sys.stderr.write('# Cannot find %s\n' % word)

  inf.close()
  ouf.close()
  if opt == 1:
    inf_We.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.word_list_file, args.out_file, args.opt)
