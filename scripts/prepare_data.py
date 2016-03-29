#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Mon Oct 13 01:22:15 PDT 2014

"""
"""

usage = 'Convert from text to integer format.' 

### Module imports ###
import sys
import os
import argparse # option parsing
import re # regular expression
import codecs
import text
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
  parser.add_argument('--vocab_file', dest='vocab_file', type=str, default='', help='vocab file (default=\'\')')
  parser.add_argument('--freq', dest='freq', type=int, default=-1, help='freq cutoff, keep words if >= (default=-1, i.e., no cutoff)')
  parser.add_argument('--size', dest='size', type=int, default=-1, help='vocab size cutoff (default=-1, i.e., no cutoff)')
  
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

def process_files(in_file, out_file, vocab_file, freq, size):
  """
  Read data from in_file, and output to out_file
  """

  sys.stderr.write('# in_file = %s, out_file = %s, freq=%d, size=%d\n' % (in_file, out_file, freq, size))
  if vocab_file=='':
    if size!=-1:
      vocab_file = in_file + '.vocab.' + str(size)
    else:
      vocab_file = in_file + '.vocab.f' + str(freq)

  # load vocab
  unk = '<unk>'
  (words, vocab_map, vocab_size) = text.get_vocab(in_file, vocab_file, freq, size, unk=unk)
  unk_id = str(vocab_map[unk])
  sys.stderr.write('# vocab_size=%d, unk_id=%s\n' % (vocab_size, unk_id))

  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
  inf = codecs.open(in_file, 'r', 'utf-8')
  ouf = codecs.open(out_file, 'w', 'utf-8')
  token_count = 0
  unk_count = 0
  for line in inf:
    indices = []
    for token in re.split('\s+', line.strip()):
      token_count += 1
      if token in vocab_map:
        indices.append(str(vocab_map[token]))
      else:
        indices.append(unk_id)
        unk_count += 1

    ouf.write('%s\n' % ' '.join(indices))
    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)

  sys.stderr.write('Done! Num lines = %d, num tokens = %d, num unks = %d, coverage = %.2f%% \n' % (line_id, token_count, unk_count, (token_count-unk_count)*100.0/token_count))

  inf.close()
  ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.out_file, args.vocab_file, args.freq, args.size)
