#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Sat Oct  3 09:50:47 PDT 2015

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
  parser.add_argument('src_file', metavar='src_file', type=str, help='input file') 
  parser.add_argument('ref_file', metavar='ref_file', type=str, help='ref file') 
  parser.add_argument('out_file', metavar='out_file', type=str, help='output file') 
  #parser.add_argument('trans_file', metavar='trans_file', type=str, help='trans file') 
  parser.add_argument('--trans', dest='trans_files', type=str, nargs='+', help='translation files')

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

def process_files(src_file, ref_file, out_file, trans_files):
  """
  Read data from src_file, and output to out_file
  """

  sys.stderr.write('# src_file = %s, out_file = %s\n' % (src_file, out_file))
  # input
  sys.stderr.write('# Input from %s.\n' % (src_file))
  src_inf = codecs.open(src_file, 'r', 'utf-8')
  ref_inf = codecs.open(ref_file, 'r', 'utf-8')
  #trans_inf = codecs.open(trans_file, 'r', 'utf-8')
  trans_infs = [codecs.open(trans_file, 'r', 'utf-8') for trans_file in trans_files]

  # output
  sys.stderr.write('Output to %s\n' % out_file)
  check_dir(out_file)
  ouf = codecs.open(out_file, 'w', 'utf-8')

  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (src_file))
  for line in src_inf:
    src_line = clean_line(line)
    ref_line = clean_line(ref_inf.readline())
    #trans_line = clean_line(trans_inf.readline())
    #ouf.write('# Sent %d\n  source:\t%s\n  human:\t%s\n  trans:\t%s\n' % (line_id, src_line, ref_line, trans_line))
    ouf.write('# Sent %d\n  source:\t%s\n  human:\t%s\n' % (line_id, src_line, ref_line))
    i = 0
    for trans_inf in trans_infs:
      i += 1
      trans_line = clean_line(trans_inf.readline())
      ouf.write('  trans %d:\t%s\n' % (i, trans_line))

    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)

  sys.stderr.write('Done! Num lines = %d\n' % line_id)

  src_inf.close()
  ref_inf.close()
  trans_inf.close()
  ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.src_file, args.ref_file, args.out_file, args.trans_files)

#  if src_file == '':
#    sys.stderr.write('# Input from stdin.\n')
#    inf = sys.stdin 
#  else:
#  if out_file == '':
#    sys.stderr.write('# Output to stdout.\n')
#    ouf = sys.stdout
#  else:
 
