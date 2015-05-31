#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, 2015

"""
"""

usage = 'Post processing translations e.g., replace <unk>' 
debug = True 

### Module imports ###
import sys
import os
import argparse # option parsing
import re # regular expression
import codecs
import text
### Function declarations ###
def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """
  
  parser = argparse.ArgumentParser(description=usage) # add description
  # positional arguments
  parser.add_argument('src_file', metavar='src_file', type=str, help='src file') 
  parser.add_argument('tgt_file', metavar='tgt_file', type=str, help='src unk file') 
  parser.add_argument('align_file', metavar='align_file', type=str, help='input cns directory to download decoded sents') 
  parser.add_argument('dict_file', metavar='dict_file', type=str, help='dict file') 
  parser.add_argument('ref_file', metavar='ref_file', type=str, help='ref file') 
  parser.add_argument('out_file', metavar='out_file', type=str, help='output file') 

  # optional arguments
  parser.add_argument('-o', '--option', dest='opt', type=int, default=0, help='0 -- copying unk, 1 -- alignment positions, 2 -- single unk handling (default=0), 3 -- alignment positions <unk_-1>, <unk_0>, <unk_1>, etc.')
  parser.add_argument('--reverse_alignment', dest='is_reverse_alignment', action='store_true', help='reverse alignment (tgtId-srcId) instead of srcId-tgtId')
  
  args = parser.parse_args()
  return args

def check_dir(out_file):
  dir_name = out_file #os.path.dirname(out_file)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def execute(cmd):
  sys.stderr.write('# Executing: %s\n' % cmd)
  os.system(cmd)

def load_dict(dict_file):
  inf = codecs.open(dict_file, 'r', 'utf-8')
  line_id = 0
  dict_map = {}
  prob_map = {}
  for line in inf:
    tokens = re.split('\s+', line.strip())
    src_word = tokens[0]
    tgt_word = tokens[1]
    prob = float(tokens[2])
    if (src_word not in dict_map) or (prob > prob_map[src_word]):
      dict_map[src_word] = tgt_word
      prob_map[src_word] = prob
    line_id += 1
    if line_id % 100000 == 0:
      sys.stderr.write(' (%d) ' % line_id)
  sys.stderr.write('  Done! Num lines = %d\n' % line_id)
  inf.close()
  return dict_map

def post_process(src_line, src_unk_line, tgt_unk_line, dict_map, opt):
  """
  Do word-word translation of unk tokens
  """
  global debug
  
  unk_pattern = '^<unk\d+>$' # opt 0
  align_pattern = '^<p_\S+>$' # opt 1: <p_n> <p_-1> <p_1>
  pos_pattern = '^<p_(.*\d+)>$' # opt 1: <p_-1> <p_1>
  unk_pos_pattern = '^<unk_(.+)>$' # opt 3: <unk_-1>, <unk_0>, <unk_1>
  unk_sym = '<unk>' # opt 1 
  
  # tgt
  tgt_unk_tokens = re.split('\s+', tgt_unk_line)
  if tgt_unk_tokens[-1] == '</s>' or tgt_unk_tokens[-1] == '<<<EOS>>>': # eos
    del tgt_unk_tokens[-1]
  if opt==2: # replace <unk> with <unk0> <unk1> etc
    count = 0
    for ii in xrange(len(tgt_unk_tokens)):
      if tgt_unk_tokens[ii] == '<unk>':
        tgt_unk_tokens[ii] = '<unk' + str(count) + '>'
        count += 1

  # map from src unk to src orig tokens
  src_tokens = re.split('\s+', src_line)
  if src_tokens[-1] == '</s>' or src_tokens[-1] == '<<<EOS>>>':
    del src_tokens[-1]

  if src_unk_line != '':
    src_unk_tokens = re.split('\s+', src_unk_line)
    if src_unk_tokens[-1] == '</s>' or src_unk_tokens[-1] == '<<<EOS>>>':
      del src_unk_tokens[-1]
    if opt==2: # replace <unk> with <unk0> <unk1> etc
      count = 0
      for ii in xrange(len(src_unk_tokens)):
        if src_unk_tokens[ii] == '<unk>':
          src_unk_tokens[ii] = '<unk' + str(count) + '>'
          count += 1
      opt=0 # now do copying
  else:
    src_unk_tokens = []
    

  src_len = len(src_tokens)
  #assert src_len==len(src_unk_tokens), 'Differ:\n  %s\n  %s\n' % (src_line, src_unk_line)

  
  # process tgt
  count = 0
  tgt_tokens = []
  count_unk_no_align = 0
  if opt==0: # unk copying
    unk_map = {}
    src_unk_count = 0
    for ii in xrange(src_len):
      if re.search(unk_pattern, src_unk_tokens[ii]):
        if src_tokens[ii] in dict_map: # there's a word-word translation
          trans = dict_map[src_tokens[ii]]
        else: # identity copy
          trans = src_tokens[ii]
        src_unk_count += 1
        unk_map[src_unk_tokens[ii]] = trans
    
    if debug==True and src_unk_count>0:
      sys.stderr.write('opt=%d\n' % opt)
      sys.stderr.write("src: %s\n" % ' '.join(src_tokens))
      sys.stderr.write("src unk: %s\n" % ' '.join(src_unk_tokens))
      sys.stderr.write("tgt unk: %s\n" % ' '.join(tgt_unk_tokens))

    new_tgt_unk_tokens = tgt_unk_tokens
    for tgt_token in tgt_unk_tokens:
      assert tgt_token != '</s>' and tgt_token != '<<<EOS>>>'
      if re.search(unk_pattern, tgt_token) and tgt_token in unk_map: # unk token
        tgt_token = unk_map[tgt_token]
        count += 1
      tgt_tokens.append(tgt_token)

  elif opt==1: # alignment handling
    new_tgt_unk_tokens = []

    # tgt_unk_tokens is expected to contains pairs of (word, position)
    if len(tgt_unk_tokens) % 2 != 0:
      sys.stderr.write('! Odd number of words: %s\n' % ' '.join(tgt_unk_tokens))
      sys.exit(1)
      #del tgt_unk_tokens[-1]

    tgt_len = len(tgt_unk_tokens) / 2
    for pos in xrange(tgt_len):
      word_token = tgt_unk_tokens[2*pos]
      pos_token = tgt_unk_tokens[2*pos+1]
      if re.search(align_pattern, word_token)!=None:
        sys.stderr.write('incorrect word: %s\n' % (tgt_unk_line))
        continue
      if re.search(align_pattern, pos_token)==None:
        sys.stderr.write('incorrect pos: %s\n' % (tgt_unk_line))
        continue

      new_tgt_unk_tokens.append(word_token)

      if word_token == unk_sym: # unk symbol, try to copy
        m = re.match(pos_pattern, pos_token)
        if m==None: # no alignment
          count_unk_no_align += 1
        else: # ah, there's an alignment to follow
          offset = int(m.group(1)) # tgt_pos-src_pos
          src_pos = pos-offset
          if src_pos>=0 and src_pos<src_len: # within boundary
            count += 1
            src_token = src_tokens[src_pos]
            if src_token in dict_map: # there's a word-word translation
              word_token = dict_map[src_token]
            else: # identity copy
              word_token = src_token

            if debug==True>0:
              sys.stderr.write('  translate: %s -> %s\n' % (src_token, word_token))
      tgt_tokens.append(word_token)
  elif opt==3: # alignment handling <unk_-1> <unk_0> <unk_1>
    new_tgt_unk_tokens = []

    tgt_len = len(tgt_unk_tokens)
    for pos in xrange(tgt_len):
      word_token = tgt_unk_tokens[pos]
      new_tgt_unk_tokens.append(word_token)

      m = re.match(unk_pos_pattern, word_token)
      if m != None: # there's an alignment to follow
        offset = int(m.group(1)) # tgt_pos-src_pos
        src_pos = pos-offset
        if src_pos>=0 and src_pos<src_len: # within boundary
          count += 1
          src_token = src_tokens[src_pos]
          if src_token in dict_map: # there's a word-word translation
            word_token = dict_map[src_token]
          else: # identity copy
            word_token = src_token

          if debug==True>0:
            sys.stderr.write('  translate: %s -> %s\n' % (src_token, word_token))
 
      tgt_tokens.append(word_token) 
  tgt_line = ' '.join(tgt_tokens)
  if debug==True and count>0:
    debug = False
    sys.stderr.write("tgt: %s\n" % tgt_line)
  return (tgt_line, new_tgt_unk_tokens, count)

def process_files(align_file, src_file, tgt_file, ref_file, dict_file, out_file, opt, is_reverse_alignment):
  """
  """
  src_inf = codecs.open(src_file, 'r', 'utf-8')
  tgt_inf = codecs.open(tgt_file, 'r', 'utf-8')
  align_inf = codecs.open(align_file, 'r', 'utf-8')
  is_ref = 0
  if ref_file != '':
    ref_inf = codecs.open(ref_file, 'r', 'utf-8')
    is_ref = 1

  # load dict
  dict_map = load_dict(dict_file)

  # out_file
  ouf = codecs.open(out_file, 'w', 'utf-8')

  # post process
  unk = '<unk>'
  line_id = 0
  debug = 1
  unk_count = 0
  dictionary_count = 0
  identity_count = 0
  for src_line in src_inf:
    src_line = src_line.strip()
    tgt_line = tgt_inf.readline().strip()
    src_tokens = re.split('\s+', src_line)
    tgt_tokens = re.split('\s+', tgt_line)
    if is_ref:
      ref_line = ref_inf.readline().strip()

    # get alignment
    align_line = align_inf.readline().strip()
    if is_reverse_alignment==True: # reversed alignment tgtId-srcId
      (t2s, s2t) = text.aggregate_alignments(align_line)
    else: # normal alignment srcId-tgtId
      (s2t, t2s) = text.aggregate_alignments(align_line)
     
    new_tgt_tokens = []
    debug_count = 0
    debug_str = ''
    for tgt_pos in xrange(len(tgt_tokens)):
      tgt_token = tgt_tokens[tgt_pos]
      if tgt_tokens[tgt_pos] == unk:
        unk_count = unk_count + 1
        if tgt_pos in t2s: # aligned unk
          debug_count = debug_count + 1
          src_token = src_tokens[t2s[tgt_pos][0]]
          if src_token in dict_map: # there's a word-word translation
            tgt_token = dict_map[src_token]
            dictionary_count = dictionary_count + 1
            if debug:
              debug_str = debug_str + "dict: " + src_token + " -> " + tgt_token + '\n'
          else: # identity copy
            tgt_token = src_token
            identity_count = identity_count + 1

            if debug:
              debug_str = debug_str + "iden: " + src_token + " -> " + tgt_token + '\n'
      new_tgt_tokens.append(tgt_token)

    out_line = ' '.join(new_tgt_tokens)
    ouf.write('%s\n' % out_line)

    # debug info
    if debug_count>0 and debug == 1:
      sys.stderr.write('# example %d\nsrc: %s\ntgt: %s\nalign: %s\n%sout: %s\n' % (line_id, src_line, tgt_line, align_line, debug_str, out_line))
      if is_ref:
        sys.stderr.write('ref: %s\n' % ref_line)
      #debug = 0

    line_id += 1   # concat results

  src_inf.close()
  tgt_inf.close()
  align_inf.close()
  ouf.close()
  sys.stderr.write('# num sents = %d, unk count=%d, dictionary_count=%d, identity_count=%d\n' % (line_id, unk_count, dictionary_count, identity_count))

  # evaluating 
  if is_ref:
    script_dir = os.path.dirname(sys.argv[0]) 
    sys.stderr.write('# Before post process\n')
    os.system(script_dir + '/multi-bleu.perl ' + ref_file + ' < ' + tgt_file)
    sys.stderr.write('# After post process\n')
    os.system(script_dir + '/multi-bleu.perl ' + ref_file + ' < ' + out_file)

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.align_file, args.src_file, args.tgt_file, args.ref_file, args.dict_file, args.out_file, args.opt, args.is_reverse_alignment)
