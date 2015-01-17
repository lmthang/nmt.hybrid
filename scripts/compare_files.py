#!/usr/bin/env python2.7
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Sat Jan 17 09:24:15 PST 2015

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
  parser.add_argument('in_file1', metavar='in_file1', type=str, help='input file') 
  parser.add_argument('in_file2', metavar='in_file2', type=str, help='output file') 

  # optional arguments
  parser.add_argument('-o', '--option', dest='opt', type=int, default=0, help='option (default=0)')
  
  args = parser.parse_args()
  return args

def check_dir(in_file2):
  dir_name = os.path.dirname(in_file2)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def clean_line(line):
  """
  Strip leading and trailing spaces
  """

  line = re.sub('(^\s+|\s$)', '', line);
  return line

def process_files(in_file1, in_file2):
  """
  Read data from in_file1, and output to in_file2
  """

  sys.stderr.write('# in_file1 = %s, in_file2 = %s\n' % (in_file1, in_file2))
  # in_file1
  sys.stderr.write('# Input from %s.\n' % (in_file1))
  inf1 = codecs.open(in_file1, 'r', 'utf-8')
  
  # in_file2
  sys.stderr.write('# Input from %s.\n' % (in_file2))
  inf2 = codecs.open(in_file2, 'r', 'utf-8')
  
  sys.stderr.write('# Processing file %s ...\n' % (in_file1))
  inf1_map = {}
  line_id = 0
  for line in inf1:
    if line in inf1_map:
      sys.stderr.write('! Duplicate %s' % line)
    inf1_map[line] = 1

    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)
  sys.stderr.write('Done! Num lines = %d\n' % line_id)
  inf1.close()
 
  sys.stderr.write('# Processing file %s ...\n' % (in_file2))
  inf2_map = {}
  line_id = 0
  for line in inf2:
    if line in inf2_map:
      sys.stderr.write('! Duplicate %s' % line)
    inf2_map[line] = 1

    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)
  sys.stderr.write('Done! Num lines = %d\n' % line_id)
  inf2.close()

  for line in inf1_map:
    if line not in inf2_map:
      sys.stderr.write('! Cannot find line in inf2_map: %s' % line)
  for line in inf2_map:
    if line not in inf1_map:
      sys.stderr.write('! Cannot find line in inf1_map: %s' % line)

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file1, args.in_file2)
