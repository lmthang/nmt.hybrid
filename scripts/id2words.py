#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Mon Mar  9 15:56:02 PDT 2015

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
  parser.add_argument('word_file', metavar='word_file', type=str, help='word file') 
  parser.add_argument('id_file', metavar='id_file', type=str, help='input file') 
  parser.add_argument('out_file', metavar='out_file', type=str, help='output preordered words') 

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

def process_files(word_file, id_file, out_file):
  """
  Read data from id_file, and output to out_file
  """

  sys.stderr.write('# word_file = %s, id_file = %s, out_file = %s\n' % (word_file, id_file, out_file))
  # input
  sys.stderr.write('# Input from %s, %s.\n' % (word_file, id_file))
  word_inf = codecs.open(word_file, 'r', 'utf-8')
  id_inf = codecs.open(id_file, 'r', 'utf-8')

  # output
  sys.stderr.write('Output to %s\n' % out_file)
  check_dir(out_file)
  ouf = codecs.open(out_file, 'w', 'utf-8')

  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (id_file))
  for line in word_inf:
    line = clean_line(line)
    tokens = re.split('\s+', line)
    num_words = len(tokens)
    id_line = clean_line(id_inf.readline())
    indices = [int(x) for x in id_line.split()]
    incorrect = 0
    reordered_words = []
    for index in indices:
      if index<0 or index>=num_words:
        incorrect = incorrect + 1
        continue
      else:
        reordered_words.append(tokens[index])

    line_id = line_id + 1
    ouf.write('%s\n' % ' '.join(reordered_words))
    if incorrect>0:
      sys.stderr.write('# word order sent %d, len %d: %s\t%s\n' % (line_id, num_words, line, id_line))

    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)

  sys.stderr.write('Done! Num lines = %d\n' % line_id)

  word_inf.close()
  ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.word_file, args.id_file, args.out_file)

#  if id_file == '':
#    sys.stderr.write('# Input from stdin.\n')
#    word_inf = sys.stdin 
#  else:
#  if out_file == '':
#    sys.stderr.write('# Output to stdout.\n')
#    ouf = sys.stdout
#  else:
 
