#!/usr/bin/env python

"""
"""

usage='Compute word-to-word dictionary'

### Module imports ###
import sys
import os
import argparse # option parsing
import re # regular expression
import codecs
import random
import math

import text

### Function declarations ###
def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """

  parser = argparse.ArgumentParser(description=usage) # add description
  # positional arguments
  parser.add_argument('in_prefix', metavar='in_prefix', type=str, help='input prefix') 
  parser.add_argument('src_lang', metavar='src_lang', type=str, help='src language') 
  parser.add_argument('tgt_lang', metavar='tgt_lang', type=str, help='tgt language') 
  parser.add_argument('out_prefix', metavar='out_prefix', type=str, help='output prefix') 

  # optional arguments
  parser.add_argument('--src_vocab_size', dest='src_vocab_size', type=int, default=-1, help='src vocab size')
  parser.add_argument('--tgt_vocab_size', dest='tgt_vocab_size', type=int, default=-1, help='tgt vocab size')
  parser.add_argument('-o', '--option', dest='opt', type=int, default=0, help='option: 0 -- normal alignment, 1 -- reverse alignment (tgtId-srcId) (default=0)')
  parser.add_argument('-f', '--freq', dest='freq', type=int, default=0, help='freq (default=5)')
  
  args = parser.parse_args()
  return args

def check_dir(out_prefix):
  dir_name = os.path.dirname(out_prefix)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def process_files(in_prefix, src_lang, tgt_lang, out_prefix, freq, opt, src_vocab_size, tgt_vocab_size, unk_symbol='<unk>'):
  """
  """
  
  # input
  sys.stderr.write('# Input from %s.*\n' % (in_prefix))
  src_file = in_prefix + '.' + src_lang
  src_inf = codecs.open(src_file, 'r', 'utf-8')
  tgt_file = in_prefix + '.' + tgt_lang
  tgt_inf = codecs.open(tgt_file, 'r', 'utf-8')
  align_inf = codecs.open(in_prefix + '.align', 'r', 'utf-8')

  if src_vocab_size>0:
    src_vocab_file = in_prefix + '.' + src_lang + '.vocab.' + str(src_vocab_size)
  elif freq>0:
    src_vocab_file = in_prefix + '.' + src_lang + '.vocab.f' + str(freq)
  (src_words, src_vocab_map, src_vocab_size) = text.get_vocab(src_file, src_vocab_file, freq, src_vocab_size, unk_symbol)
  
  if tgt_vocab_size>0:
    tgt_vocab_file = in_prefix + '.' + tgt_lang + '.vocab.' + str(tgt_vocab_size)
  elif freq>0:
    tgt_vocab_file = in_prefix + '.' + tgt_lang + '.vocab.f' + str(freq)  
  (tgt_words, tgt_vocab_map, tgt_vocab_size) = text.get_vocab(tgt_file, tgt_vocab_file, freq, tgt_vocab_size, unk_symbol)
  
  # process corpus
  line_id = 0
  debug = True
  bi_counts = {} # bi_counts[src_id][tgt_id]
  src_counts = {}
  tgt_counts = {}
  total_count = 0 # total alignment links
  for src_line in src_inf:
    src_line = src_line.strip()
    tgt_line = tgt_inf.readline().strip()
    src_tokens = re.split('\s+', src_line)
    tgt_tokens = re.split('\s+', tgt_line)
    if opt==1: # reversed alignment tgtId-srcId
      (t2s, s2t) = text.aggregate_alignments(align_inf.readline())
    else: # normal alignment srcId-tgtId
      (s2t, t2s) = text .aggregate_alignments(align_inf.readline())

    # process alignments
    for tgt_pos in t2s.keys():
      for src_pos in t2s[tgt_pos]:
        # same word
        src_token = src_tokens[src_pos]
        tgt_token = tgt_tokens[tgt_pos]
        if src_token in src_vocab_map and tgt_token in tgt_vocab_map: # both known
          src_id = src_vocab_map[src_token]
          tgt_id = tgt_vocab_map[tgt_token]
          if src_id not in bi_counts:
            bi_counts[src_id] = {}
            src_counts[src_id] = 0
          if tgt_id not in tgt_counts:
            tgt_counts[tgt_id] = 0
          if tgt_id not in bi_counts[src_id]:
            bi_counts[src_id][tgt_id] = 0
          
          # update
          bi_counts[src_id][tgt_id] += 1
          src_counts[src_id] += 1
          tgt_counts[tgt_id] += 1
          total_count += 1

    line_id = line_id + 1
    if (line_id % 100000 == 0):
      sys.stderr.write(' (%d) ' % line_id)
  sys.stderr.write('  num lines=%d, total links=%d\n' % (line_id, total_count))

  # output
  check_dir(out_prefix)
  dict_file = out_prefix + '.' + src_lang + '-' + tgt_lang + '.dict'
  dict_ouf = codecs.open(dict_file, 'w', 'utf-8')
  sys.stderr.write('# Output to %s*\n' % dict_file)

  # compute src_probs
  src_probs = {}
  for src_id in src_counts.keys():
    src_probs[src_id] = float(src_counts[src_id])/float(total_count)

  # compute tgt_probs
  tgt_probs = {}
  for tgt_id in tgt_counts.keys():
    tgt_probs[tgt_id] = float(tgt_counts[tgt_id])/float(total_count)

  # compute joint prob
  for src_id in bi_counts.keys():
    for tgt_id in bi_counts[src_id].keys():
      bi_count = bi_counts[src_id][tgt_id]
      if bi_count<10: continue
      p_src_given_tgt = float(bi_count)/float(tgt_counts[tgt_id])
      p_tgt_given_src = float(bi_count)/float(src_counts[src_id])
      
      # normalized pmi
      p_src_tgt = float(bi_count)/float(total_count) # joint
      p_src = src_probs[src_id]
      p_tgt = tgt_probs[tgt_id]
      pmi = math.log(p_src_tgt/(p_src*p_tgt))
      npmi = - pmi / math.log(p_src_tgt) 
  
      # print
      src_token = src_words[src_id]
      tgt_token = tgt_words[tgt_id]
      dict_ouf.write('%s %s %g %g %g %g %g\n' % (src_token, tgt_token, p_tgt_given_src, p_src_given_tgt, (p_src_given_tgt+p_tgt_given_src)/2, pmi, npmi))
      #dict_ouf.write('%s %s %g\n' % (src_token, tgt_token, (p_src_given_tgt+p_tgt_given_src)/2))

  #text.write_vocab(out_prefix + '.vocab.' + src_lang, src_words)
  #text.write_vocab(out_prefix + '.vocab.' + tgt_lang, tgt_words)

  src_inf.close()
  tgt_inf.close()
  align_inf.close()

  dict_ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_prefix, args.src_lang, args.tgt_lang, args.out_prefix, args.freq, args.opt, args.src_vocab_size, args.tgt_vocab_size)

