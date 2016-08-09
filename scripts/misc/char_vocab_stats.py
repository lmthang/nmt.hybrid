#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Tue Jan 19 15:52:54 PST 2016

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
  #parser.add_argument('out_file', metavar='out_file', type=str, help='output file') 

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

def process_files(in_file):
  """
  Read data from in_file, and output to out_file
  """

  sys.stderr.write('# in_file = %s\n' % (in_file))
  # input
  sys.stderr.write('# Input from %s.\n' % (in_file))
  inf = codecs.open(in_file, 'r', 'utf-8')

  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
  lens = []
  for line in inf:
    line = clean_line(line)
    lens.append(len(line))
    if len(line)>200:
      print line
    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)

  sys.stderr.write('Done! Num words = %d, min len = %d, max len = %d, avg = %d\n' % (line_id, min(lens), max(lens), sum(lens)/len(lens)))

  inf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file)
