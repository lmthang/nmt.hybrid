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
  parser.add_argument('--bin_size', dest='bin_size', type=int, default=10, help='bin_size: group sents by lengths of 1:bin_size, (bin_size+1):2*bin_size, etc (default=10)')
  parser.add_argument('-d', '--debug', dest='debug', action='store_true', default=False, help='enable debugging mode (default: false)') 
  
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

def process_buffer(sents, sent_lens, sent_ids, tgt_sents, align_sents, bin_size, batch_size, is_parallel):
  '''
  Process currently loaded sents
  '''

  group_count = 0
  group_sents = []
  group_sent_lens = []
  group_sent_ids = []
  tgt_group_sents = []
  align_group_sents = []

  # pick the first sent & decide bin
  sent_len = sent_lens[0]
  bin_id = (sent_len-1)/bin_size
  start_len = bin_id * bin_size + 1
  end_len = (bin_id+1)*bin_size 
  #sys.stderr.write('sent_len %d, bin_id %d, start_len %d, end_len %d\n' % (sent_len, bin_id, start_len, end_len))
  
  # go through the currently loaded sents and select those that are in the same bin
  ii = 0
  while ii < len(sents):
    if sent_lens[ii]>=start_len and sent_lens[ii]<=end_len: # in the same bin
      group_count += 1
      group_sents.append(sents[ii])
      group_sent_lens.append(sent_lens[ii])
      group_sent_ids.append(sent_ids[ii])
      
      # remove sent
      del sents[ii]
      del sent_lens[ii]
      del sent_ids[ii]

      if is_parallel:
        tgt_group_sents.append(tgt_sents[ii])
        del tgt_sents[ii]
        align_group_sents.append(align_sents[ii])
        del align_sents[ii]

      if group_count == batch_size: # select enough
        break
    else:
      ii += 1

  return (sents, sent_lens, sent_ids, group_sents, group_sent_lens, group_sent_ids, tgt_group_sents, align_group_sents, group_count, start_len, end_len)
 
def process_files(in_file, src_lang, tgt_lang, out_file, batch_size, bin_size):
  """
  Read data from in_file, and output to out_file
  """

  sys.stderr.write('# in_file = %s, src_lang = %s, tgt_lang = %s, out_file = %s, batch_size = %d, bin_size = %d\n' % (in_file, src_lang, tgt_lang, out_file, batch_size, bin_size))
   
  # IO
  is_parallel = False
  check_dir(out_file)
  if src_lang != '':
    assert tgt_lang != ''
    is_parallel = True
    inf = codecs.open(in_file + '.' + src_lang, 'r', 'utf-8')
    tgt_inf = codecs.open(in_file + '.' + tgt_lang, 'r', 'utf-8')
    align_inf = codecs.open(in_file + '.align', 'r', 'utf-8')
    ouf = codecs.open(out_file + '.' + src_lang, 'w', 'utf-8')
    tgt_ouf = codecs.open(out_file + '.' + tgt_lang, 'w', 'utf-8')
    align_ouf = codecs.open(out_file + '.align', 'w', 'utf-8')
  else:
    inf = codecs.open(in_file, 'r', 'utf-8')
    ouf = codecs.open(out_file, 'w', 'utf-8')
  
  id_ouf = codecs.open(out_file + '.id', 'w', 'utf-8')

  # buffer
  sents = []
  sent_lens = []
  sent_ids = []
  tgt_sents = []
  align_sents = []
  
  # group
  group_count = 0
  group_sents = []
  group_sent_lens = []
  group_sent_ids = []
  tgt_group_sents = []
  align_group_sents = []

  # load the first sent
  sent = clean_line(inf.readline())
  tokens = re.split('\s+', sent)
  sents.append(sent)
  sent_lens.append(len(tokens))
  sent_ids.append(0)
  line_id = 1
  if is_parallel:
    tgt_sents.append(clean_line(tgt_inf.readline()))
    align_sents.append(clean_line(align_inf.readline()))

  for sent in inf:
    while group_count == 0 or group_count == batch_size: # process buffer
      if group_count == batch_size: # output
        sys.stderr.write('# line_id=%d, [%d, %d], group count %d\n' % (line_id, start_len, end_len, group_count))
        for group_sent in group_sents:
          ouf.write('%s\n' % group_sent)
        for group_id in group_sent_ids:
          id_ouf.write('%s\n' % group_id)
        if is_parallel:
          for tgt_group_sent in tgt_group_sents:
            tgt_ouf.write('%s\n' % tgt_group_sent)
          for align_group_sent in align_group_sents:
            align_ouf.write('%s\n' % align_group_sent)

      (sents, sent_lens, sent_ids, group_sents, group_sent_lens, group_sent_ids, tgt_group_sents, align_group_sents, group_count, start_len, end_len) = process_buffer(sents, sent_lens, sent_ids, tgt_sents, align_sents, bin_size, batch_size, is_parallel)
      assert group_count>0
    assert group_count<batch_size
    assert group_count>0

    # process the new sent
    sent = clean_line(sent)
    tokens = re.split('\s+', sent)
    sent_len = len(tokens)

    if is_parallel:
      tgt_sent = clean_line(tgt_inf.readline())
      align_sent = clean_line(align_inf.readline())

    if sent_len>=start_len and sent_len<=end_len: # in the same bin
      group_count += 1
      group_sents.append(sent)
      group_sent_lens.append(sent_len)
      group_sent_ids.append(line_id)
      if is_parallel:
        tgt_group_sents.append(tgt_sent)
        align_group_sents.append(align_sent)

    else: # not in the same bin, put into the buffer
      sents.append(sent)
      sent_lens.append(sent_len)
      sent_ids.append(line_id) 
      if is_parallel:
        tgt_sents.append(tgt_sent)
        align_sents.append(align_sent)


      if len(sents)>=10000: # buffer is large (due to the fact that long sentences are hard to find), move the current group of sents to the end, reset
        sents.extend(group_sents)
        sent_lens.extend(group_sent_lens)
        sent_ids.extend(group_sent_ids)
        if is_parallel:
          tgt_sents.extend(tgt_group_sents)
          align_sents.extend(align_group_sents)

        sys.stderr.write('  line_id=%d, move %d sents in [%d, %d] to the end\n' % (line_id, group_count, start_len, end_len))
        group_count = 0

    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)
 
  if group_count>0: # move to the end
    sys.stderr.write('  line_id=%d, last group, move %d sents in [%d, %d] to the end, buffer size %d\n' % (line_id, group_count, start_len, end_len, len(sents)))
    sents.extend(group_sents)
    sent_lens.extend(group_sent_lens)
    sent_ids.extend(group_sent_ids)
    if is_parallel:
      tgt_sents.extend(tgt_group_sents)
      align_sents.extend(align_group_sents)


  # handle the remaining
  num_remain_sents = len(sents)
  count = 0 
  while len(sents)>0:
    (sents, sent_lens, sent_ids, group_sents, group_sent_lens, group_sent_ids, tgt_group_sents, align_group_sents, group_count, start_len, end_len) = process_buffer(sents, sent_lens, sent_ids, tgt_sents, align_sents, bin_size, batch_size, is_parallel)
    if len(group_sents) == batch_size or len(sents) < batch_size: 
      sys.stderr.write('# line_id=%d, [%d, %d], group count %d\n' % (line_id, start_len, end_len, group_count))
      for group_sent in group_sents:
        ouf.write('%s\n' % group_sent)
      for group_id in group_sent_ids:
        id_ouf.write('%s\n' % group_id)
      if is_parallel:
        for tgt_group_sent in tgt_group_sents:
          tgt_ouf.write('%s\n' % tgt_group_sent)
        for align_group_sent in align_group_sents:
          align_ouf.write('%s\n' % align_group_sent)
    else: # push to the end
      sents.extend(group_sents)
      sent_lens.extend(group_sent_lens)
      sent_ids.extend(group_sent_ids)

    count += len(group_sents)
    if count>= num_remain_sents: # gone through all remaining sents, relax bin_size
      bin_size *= 2
      sys.stderr.write('# Update bin_size to %d\n' % bin_size)
      count = 0
      num_remain_setns = len(sents)
      

  sys.stderr.write('Done! Num lines = %d\n' % line_id)

  inf.close()
  ouf.close()
  id_ouf.close()
  if is_parallel:
    tgt_inf.close()
    align_inf.close()
    tgt_ouf.close()
    align_ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.src_lang, args.tgt_lang, args.out_file, args.batch_size, args.bin_size)
