#!/usr/bin/env python
# Author: Minh-Thang Luong <luong.m.thang@gmail.com>, created on Sun Nov  2 17:34:30 EST 2014

"""
Module docstrings.
"""

usage = 'Group sentences of similar lengths together'

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
  parser.add_argument('in_file', metavar='in_file', type=str, help='input file (when src_lang & tgt_lang are specified, we assume files $in_file.($src_lang|$tgt_lang|align) exist.') 
  parser.add_argument('out_file', metavar='out_file', type=str, help='output file') 

  # optional arguments
  parser.add_argument('--src_lang', dest='src_lang', type=str, default='', help='src lang (default=\'\')') 
  parser.add_argument('--tgt_lang', dest='tgt_lang', type=str, default='', help='tgt lang (default=\'\')') 
  parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size (default=128)') 
  parser.add_argument('--max_len', dest='max_len', type=int, default=100, help='sentences with lengths outside of this threshold is put into a separate group (default=100)') 
  parser.add_argument('--num_bins', dest='num_bins', type=int, default=6, help='num_bins: splits sentences of lens [1, maxLen] into (num_bins-1) bins, and the rest of the sentences into the last bin (default=6)')
  parser.add_argument('-d', '--debug', dest='debug', action='store_true', default=False, help='enable debugging mode (default: false)') 
  
  args = parser.parse_args()
  return args

def check_dir(out_file):
  dir_name = os.path.dirname(out_file)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def load_entire_file(in_file, num_bins, max_len):
  inf = codecs.open(in_file, 'r', 'utf-8')
  sents = []

  bin_len = max_len / (num_bins-1) # the first (num_bins-1) will have sentences with lens [1 max_len], the rest goes into the last bin
  sys.stderr.write('num_bins=%d, max_len=%d, bin_len=%d\n' % (num_bins, max_len, bin_len))
  bin_lists = [] # bin_lists[i]: list of sentence ids in that bin
  for ii in xrange(num_bins):
    bin_lists.append([])

  line_id = 0
  num_tokens = 0
  sys.stderr.write('# Loading file %s' % in_file)
  sent_max_len = 0
  for line in inf:
    sent_len = len(line.split())
    sents.append(line)
    bin_id = (sent_len-1)/bin_len
    if bin_id >= num_bins:
      bin_id = num_bins-1
    bin_lists[bin_id].append(line_id)
    if sent_len > sent_max_len:
      sent_max_len = sent_len
    #sys.stderr.write('%d\t%d\t%s' % (sent_len, bin_id, line))

    num_tokens += sent_len
    line_id = line_id + 1
    if (line_id % 1000000 == 0):
      sys.stderr.write(' (%d) ' % line_id)
  
  sys.stderr.write('Done! Num lines = %d, num tokens = %d, sent max len = %d.\n' % (line_id, num_tokens, sent_max_len))
  inf.close()

  return (sents, bin_lists, bin_len, sent_max_len) 
   
def process_files(in_file, src_lang, tgt_lang, out_file, batch_size, num_bins, max_len):
  """
  Read data from in_file, and output to out_file
  """

  sys.stderr.write('# in_file = %s, src_lang = %s, tgt_lang = %s, out_file = %s, batch_size = %d, num_bins = %d, max_len = %d\n' % (in_file, src_lang, tgt_lang, out_file, batch_size, num_bins, max_len))
  check_dir(out_file)
   
  # IO
  is_parallel = False
  is_align = False
  tgt_sents = []
  align_sents = []
  if src_lang != '':
    assert tgt_lang != ''
    is_parallel = True

    # src
    (sents, bin_lists, bin_len, sent_max_len) = load_entire_file(in_file + '.' + src_lang, num_bins, max_len)
    ouf = codecs.open(out_file + '.' + src_lang, 'w', 'utf-8')

    # tgt
    tgt_file = in_file + '.' + tgt_lang
    sys.stderr.write('# Loading tgt file %s ...' % tgt_file)
    tgt_inf = codecs.open(tgt_file, 'r', 'utf-8')
    tgt_sents = tgt_inf.readlines()
    sys.stderr.write('# Done! Num lines = %d\n' % len(tgt_sents))
    tgt_inf.close()
    tgt_ouf = codecs.open(out_file + '.' + tgt_lang, 'w', 'utf-8')

    # align
    align_file = in_file + '.align'
    if os.path.isfile(align_file): 
      sys.stderr.write('# Loading align file %s ...' % align_file)
      align_inf = codecs.open(align_file, 'r', 'utf-8')
      align_sents = align_inf.readlines()
      sys.stderr.write('# Done! Num lines = %d\n' % len(align_sents))
      align_inf.close()
      align_ouf = codecs.open(out_file + '.align', 'w', 'utf-8')
      is_align = True
  else:
    (sents, bin_lists, bin_len, sent_max_len) = load_entire_file(in_file, num_bins, max_len)
    ouf = codecs.open(out_file, 'w', 'utf-8')
  id_ouf = codecs.open(out_file + '.id', 'w', 'utf-8')

  opt = 0
  if opt==0: # output all short batches then longer
    end_points = []
    
    # print batches
    for ii in xrange(num_bins):
      start_len = ii*bin_len + 1
      end_len = (ii+1)*bin_len
      if ii==(num_bins-1):
        end_len = sent_max_len
      num_sents = len(bin_lists[ii])

      num_batchs = num_sents / batch_size
      sys.stderr.write('# [%d, %d]: num sents %d, num_batchs %d\n' % (start_len, end_len, num_sents, num_batchs))
      #ouf.write('# [%d, %d]: num sents %d, num_batchs %d\n' % (start_len, end_len, num_sents, num_batchs))
      for jj in xrange(num_batchs*batch_size):
        index = bin_lists[ii][jj]
        ouf.write('%s' % sents[index])
        id_ouf.write('%d\n' % index)
        if is_parallel:
          tgt_ouf.write('%s' % tgt_sents[index])
          if is_align:
            align_ouf.write('%s' % align_sents[index])

      end_points.append(num_batchs*batch_size)
    
    # print remained sents
    for ii in xrange(num_bins):
      sys.stderr.write('# remained sents = %d\n' % (len(bin_lists[ii])-end_points[ii])) 
      #ouf.write('# remained sents = %d\n' % (len(bin_lists[ii])-end_points[ii])) 
      for jj in xrange(end_points[ii], len(bin_lists[ii])):
        index = bin_lists[ii][jj]
        ouf.write('%s' % sents[index])
        id_ouf.write('%d\n' % index)
        if is_parallel:
          tgt_ouf.write('%s' % tgt_sents[index])
          if is_align:
            align_ouf.write('%s' % align_sents[index])

  ouf.close()
  id_ouf.close()
  if is_parallel:
    if is_align:
      align_ouf.close()
    tgt_ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.src_lang, args.tgt_lang, args.out_file, args.batch_size, args.num_bins, args.max_len)
