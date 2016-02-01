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
  parser.add_argument('char_size', metavar='char_size', type=int, help='number of char vocab size') 

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

def process_files(in_file, out_prefix, char_size):
  """
  Read data from in_file, and output to out_prefix
  """

  sys.stderr.write('# in_file = %s, out_prefix = %s, char_size = %d\n' % (in_file, out_prefix, char_size))

  # input
  sys.stderr.write('# Input from %s.\n' % (in_file))
  inf = codecs.open(in_file, 'r', 'utf-8')

  # output
  check_dir(out_prefix)
  dict_out_file = out_prefix + '.char.dict'
  map_out_file = out_prefix + '.char.map'
  filtered_out_file = out_prefix + '.vocab'
  sys.stderr.write('Output to %s, %s, %s\n' % (dict_out_file, map_out_file, filtered_out_file))
  dict_ouf = codecs.open(dict_out_file, 'w', 'utf-8')
  map_ouf = codecs.open(map_out_file, 'w', 'utf-8')
  filtered_ouf = codecs.open(filtered_out_file, 'w', 'utf-8')


  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
  vocab = []
  char_counts = {} # count how many times a character appears in words
  for line in inf:
    word = clean_line(line)
    vocab.append(word)
    for char in word:
      if char not in char_counts:
        char_counts[char] = 0
      char_counts[char] += 1
  
    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%dK) ' % (line_id/1000))
  inf.close()
  sys.stderr.write('Done! Num words = %d, num chars = %d\n' % (line_id, len(char_counts)))
 
  # select the top chars
  char_vocab_out_file = out_prefix + '.char.vocab'
  char_vocab_full_out_file = out_prefix + '.char.vocab.full'
  sys.stderr.write('# Selecting the top %d chars, output to %s, %s\n' % (char_size, char_vocab_out_file, char_vocab_full_out_file))
  top_chars = sorted(char_counts.iteritems(), key=lambda x: x[1], reverse=True)
  char_ouf = codecs.open(char_vocab_out_file, 'w', 'utf-8')
  char_full_ouf = codecs.open(char_vocab_full_out_file, 'w', 'utf-8')
  char_dict = {}
  char_map = {}
  chars = []
  num_chars = 0
  for (char, count) in top_chars:
    if num_chars < char_size or char == '<' or char == '>' or char == '/': # the check of <, /, > is for <unk>, <s>, </s>
      char_ouf.write('%s\n' % (char))
      char_dict[char] = 0
      char_map[char] = num_chars
      chars.append(char)
      num_chars += 1
    char_full_ouf.write('%s %d\n' % (char, count))

  # filter vocab
  sys.stderr.write('# Filtering word vocab ...\n')
  filtered_vocab_size = 0
  for word in vocab:
    char_indices = []
    all_char_known = True
    for char in word:
      if char not in char_map:
        all_char_known = False
        break
      char_dict[char] += 1
      char_indices.append(str(char_map[char]))

    # we know all chars, so add word to the filtered vocab
    if all_char_known:
      filtered_ouf.write('%s\n' % word)
      map_ouf.write('%s\n' % ' '.join(char_indices))
      filtered_vocab_size += 1 
  
  for (k, v) in sorted(char_dict.items(), key=lambda x: x[1], reverse=True):
    dict_ouf.write('%s %d\n' % (k, v))
 
  sys.stderr.write('Done! Num words %d, num chars %d, filtered_vocab_size = %d\n' % (line_id, len(char_counts), filtered_vocab_size))
    
  dict_ouf.close()
  map_ouf.close()
  filtered_ouf.close()
  char_ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.out_prefix, args.char_size)

