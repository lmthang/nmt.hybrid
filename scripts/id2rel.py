#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Sat Mar 14 01:11:08 PDT 2015

"""
Module docstrings.
"""

usage = 'Convert a permutation of absolute indices into relative indices.' 

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
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
  max_dist = 20
  for line in inf:
    line = clean_line(line)
    tokens = re.split('\s+', line)
    new_tokens = []
    for ii in xrange(len(tokens)):
      rel_dist = ii-int(tokens[ii])
      if rel_dist>max_dist or rel_dist<-max_dist:
        rel_dist=0
      new_tokens.append(str(rel_dist))

    ouf.write('%s\n' % ' '.join(new_tokens))

    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)

  sys.stderr.write('Done! Num lines = %d\n' % line_id)

  inf.close()
  ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.out_file)

#  if in_file == '':
#    sys.stderr.write('# Input from stdin.\n')
#    inf = sys.stdin 
#  else:
#  if out_file == '':
#    sys.stderr.write('# Output to stdout.\n')
#    ouf = sys.stdout
#  else:
 
