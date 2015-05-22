#!/usr/bin/env python

"""
"""

usage='Annotate unk with alignment info'

### Module imports ###
import sys
import os
import argparse # option parsing
import re # regular expression
import codecs
import random

import text
reload(sys)
sys.setdefaultencoding('utf-8')

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
  parser.add_argument('opt', metavar='opt', type=int, help='option: 0 -- annotating unks not using alignments, 1 -- annotating unks with aligned positions outputed, 2 -- annotating unks with copying information, 3 -- like 0 but use a single <unk>, 4 -- like 1 but only produce positions for unks')

  # optional arguments
  parser.add_argument('--src_output_opt', dest='src_output_opt', type=int, default=0, help='source output option: 0 -- normal, 1 -- reverse the source side, 2 -- forward src followed by reverse src, 3 -- reverse src, reverser src')
  parser.add_argument('--dict', dest='dict_file', type=str, default='', help='dict file')
  parser.add_argument('--src_vocab', dest='src_vocab_file', type=str, default='', help='src vocab file')
  parser.add_argument('--tgt_vocab', dest='tgt_vocab_file', type=str, default='', help='tgt vocab file')
  parser.add_argument('--src_vocab_size', dest='src_vocab_size', type=int, default=-1, help='src vocab size')
  parser.add_argument('--tgt_vocab_size', dest='tgt_vocab_size', type=int, default=-1, help='tgt vocab size')
  parser.add_argument('--reverse_alignment', dest='is_reverse_alignment', action='store_true', help='reverse alignment (tgtId-srcId) instead of srcId-tgtId')

  parser.add_argument('--freq', dest='freq', type=int, default=0, help='freq (default=5)')
  parser.add_argument('--window', dest='window', type=int, default=7, help='distance window (default=7)')
  parser.add_argument('--separate_output', dest='is_separate_output', action='store_true', default=False, help='output src and tgt indices into separate files (default=False)')
  parser.add_argument('--absolute', dest='is_absolute', action='store_true', default=False, help='output absolute positions instead of relative positions (default=False)')
  parser.add_argument('--separate_pos', dest='is_separate_pos', action='store_true', default=False, help='output tgt words and positions into separate files (default=False)')
  parser.add_argument('--no_eos', dest='no_eos', action='store_true', default=False, help='no eos (default=False)')
  
  args = parser.parse_args()
  return args

def check_dir(out_prefix):
  dir_name = os.path.dirname(out_prefix)

  if dir_name != '' and os.path.exists(dir_name) == False:
    sys.stderr.write('! Directory %s doesn\'t exist, creating ...\n' % dir_name)
    os.makedirs(dir_name)

def find_word_pos(tgt_pos, src_tokens, tgt_tokens, t2s, tgt_vocab_map, dict_map, window):
  tgt_token = tgt_tokens[tgt_pos]

  best_src_pos = -1
  if tgt_pos in t2s: # aligned word
    src_positions = t2s[tgt_pos]
  
    # find the best aligned src word
    best_src_prob = -1
    for src_pos in src_positions:
      if src_pos < (tgt_pos-window) or src_pos > (tgt_pos+window): continue # out of range
      if src_pos>=len(src_tokens):
        sys.stderr.write('! wrong alignment: src_pos>=len(src_tokens)\n')
        print(src_pos)
        print(src_tokens)
        print(tgt_pos)
        print(tgt_tokens)
        sys.exit(1)
      
      src_token = src_tokens[src_pos]
      prob = 0
      if src_token in dict_map and tgt_token in dict_map[src_token]:
        prob = dict_map[src_token][tgt_token] 
      if prob > best_src_prob:
        best_src_pos = src_pos
        best_src_prob = prob
    

  if tgt_token not in tgt_vocab_map: # unk target word
    tgt_token = '<unk>'

  return (tgt_token, best_src_pos)

def add_unks(words, vocab_map, vocab_size, num_unks):
  for i in  xrange(num_unks): 
    unk = '<unk' + str(i) + '>'
    (words, vocab_map, vocab_size) = text.add_word_to_vocab(unk, words, vocab_map, vocab_size)
  return (words, vocab_map, vocab_size)

def process_files(in_prefix, src_lang, tgt_lang, out_prefix, freq, is_reverse_alignment, opt, dict_file, src_vocab_file, tgt_vocab_file, src_vocab_size, tgt_vocab_size, src_output_opt, is_separate_output, is_separate_pos, no_eos, window, is_absolute, eos = '</s>', delim='*', unk_symbol='<unk>'):
  """
  """
  
  # input
  sys.stderr.write('# Input from %s.*\n' % (in_prefix))
  sys.stderr.write('# opt=%d\n' % (opt))
  
  src_file = in_prefix + '.' + src_lang
  src_inf = codecs.open(src_file, 'r', 'utf-8')
  tgt_file = in_prefix + '.' + tgt_lang
  tgt_inf = codecs.open(tgt_file, 'r', 'utf-8')
  if opt>0:
    align_inf = codecs.open(in_prefix + '.align', 'r', 'utf-8')

  # dict file
  dict_map = {}
  dict_size = 0
  if opt>0 and dict_file != '':
    (dict_map, dict_size) = text.load_dict(dict_file)

  if src_vocab_file == '':
    src_vocab_file = in_prefix + '.' + src_lang + '.vocab.' + str(src_vocab_size) 
  (src_words, src_vocab_map, src_vocab_size) = text.get_vocab(src_file, src_vocab_file, freq, src_vocab_size, unk_symbol)
  
  if tgt_vocab_file == '':
    tgt_vocab_file = in_prefix + '.' + tgt_lang + '.vocab.' + str(tgt_vocab_size)
  (tgt_words, tgt_vocab_map, tgt_vocab_size) = text.get_vocab(tgt_file, tgt_vocab_file, freq, tgt_vocab_size, unk_symbol)
  
  # output
  check_dir(out_prefix)
  if is_separate_output:
    src_id_file = out_prefix + '.id.' + src_lang 
    tgt_id_file = out_prefix + '.id.' + tgt_lang
    src_id_ouf = codecs.open(src_id_file, 'w', 'utf-8')
    tgt_id_ouf = codecs.open(tgt_id_file, 'w', 'utf-8')
    if is_separate_pos:
      tgt_pos_id_file = out_prefix + '.id.' + tgt_lang + '.pos'
      tgt_pos_id_ouf = codecs.open(tgt_pos_id_file, 'w', 'utf-8')
  else:
    src_tgt_id_file = out_prefix + '.' + src_lang + '-' + tgt_lang + '.id'
    src_tgt_id_ouf = codecs.open(src_tgt_id_file, 'w', 'utf-8')

  src_unk_file = out_prefix + '.' + src_lang
  src_ouf = codecs.open(src_unk_file, 'w', 'utf-8')
  tgt_unk_file = out_prefix + '.' + tgt_lang
  tgt_ouf = codecs.open(tgt_unk_file, 'w', 'utf-8')
    
  if opt==1:
    num_unks = 0
    #if is_absolute==False: # relative position = tgt_pos - src_pos
    #  # positions -window ... -1
    #  for i in xrange(window, 0, -1): 
    #    pos_word = '<p_' + str(-i) + '>'
    #    (tgt_words, tgt_vocab_map, tgt_vocab_size) = text.add_word_to_vocab(pos_word, tgt_words, tgt_vocab_map, tgt_vocab_size)

    #  # position 0
    #  pos_word = '<p_0>'
    #  (tgt_words, tgt_vocab_map, tgt_vocab_size) = text.add_word_to_vocab(pos_word, tgt_words, tgt_vocab_map, tgt_vocab_size)
    #  
    #  # 1 ... window
    #  for i in xrange(window): 
    #    pos_word = '<p_' + str(i+1) + '>'
    #    (tgt_words, tgt_vocab_map, tgt_vocab_size) = text.add_word_to_vocab(pos_word, tgt_words, tgt_vocab_map, tgt_vocab_size)

    #  # null alignment
    #  pos_null = '<p_n>'
    #  (tgt_words, tgt_vocab_map, tgt_vocab_size) = text.add_word_to_vocab(pos_null, tgt_words, tgt_vocab_map, tgt_vocab_size)
  elif opt==2 or opt==0:
    num_unks = 20
    (src_words, src_vocab_map, src_vocab_size) = add_unks(src_words, src_vocab_map, src_vocab_size, num_unks)
    (tgt_words, tgt_vocab_map, tgt_vocab_size) = add_unks(tgt_words, tgt_vocab_map, tgt_vocab_size, num_unks)
  elif opt==3: # for opt 3, we only use <unk>
    num_unks = 0
  elif opt==4: # for opt 4, use <unk_0>, <unk+1>, <unk-1>, etc. for the target 
    num_unks = 0

    # add relative unk position words to tgt vocab
    for i in xrange(window): 
      pos_word = '<unk_' + str(i+1) + '>'
      (tgt_words, tgt_vocab_map, tgt_vocab_size) = text.add_word_to_vocab(pos_word, tgt_words, tgt_vocab_map, tgt_vocab_size)
      pos_word = '<unk_' + str(-i-1) + '>'
      (tgt_words, tgt_vocab_map, tgt_vocab_size) = text.add_word_to_vocab(pos_word, tgt_words, tgt_vocab_map, tgt_vocab_size)

    # position 0
    pos_word = '<unk_0>'
    (tgt_words, tgt_vocab_map, tgt_vocab_size) = text.add_word_to_vocab(pos_word, tgt_words, tgt_vocab_map, tgt_vocab_size)
  else:
    sys.err.write('! Unknown opt %d\n' % opt)
    sys.exit(1)

  if src_output_opt==2 or src_output_opt==3: # add <src_split>
    src_split = '<src_split>'
    (src_words, src_vocab_map, src_vocab_size) = text.add_word_to_vocab(src_split, src_words, src_vocab_map, src_vocab_size)

  text.write_vocab(out_prefix + '.vocab.' + src_lang, src_words)
  text.write_vocab(out_prefix + '.vocab.' + tgt_lang, tgt_words)

 
  # unk stats on the target side
  #num_global_aligned_unks = 0
  #num_global_unaligned_unks = 0
 
  line_id = 0
  if is_separate_output:
    sys.stderr.write('# Output to %s* and %s*\n' % (src_id_file, tgt_id_file))
  else:
    sys.stderr.write('# Output to %s*\n' % src_tgt_id_file)

  debug = True
  align_debug = True
  
  num_global_unks = 0
  approx_align_found = 0
  approx_align_notfound = 0
  global_forw_dist = 0
  global_back_dist = 0
  num_aligns = 0
  for src_line in src_inf:
    src_line = src_line.strip()
    tgt_line = tgt_inf.readline().strip()
    src_tokens = re.split('\s+', src_line)
    tgt_tokens = re.split('\s+', tgt_line)

    if opt>0:
      align_line = align_inf.readline()
      if is_reverse_alignment==True: # reversed alignment tgtId-srcId
        (t2s, s2t) = text.aggregate_alignments(align_line)
      else: # normal alignment srcId-tgtId
        (s2t, t2s) = text.aggregate_alignments(align_line)

    ### annotate src side
    (src_unk_tokens, src_num_unk_tokens, src_num_unk_types) = text.annotate_unk(src_tokens, src_vocab_map, num_unks)
    src_unk_tokens.append(eos)

    ### annotate tgt side
    #num_aligned_unks = 0
    #forw_dist = 0
    #back_dist = 0
    if opt==0 or opt==3: # not using the alignment
      (tgt_unk_tokens, tgt_num_unk_tokens, tgt_num_unk_types) = text.annotate_unk(tgt_tokens, tgt_vocab_map, num_unks)

    elif opt==1 or opt==4: # annotating unks with aligned positions
      assert dict_size > 0
      tgt_unk_tokens = []
      #if is_absolute: # positions
      tgt_unk_positions = []

      new_tgt_tokens = []
      best_src_positions = []
      for tgt_pos in xrange(len(tgt_tokens)):
        (tgt_token, best_src_pos) = find_word_pos(tgt_pos, src_tokens, tgt_tokens, t2s, tgt_vocab_map, dict_map, window)
        new_tgt_tokens.append(tgt_token)
        best_src_positions.append(best_src_pos)

      for tgt_pos in xrange(len(tgt_tokens)):
        tgt_token = new_tgt_tokens[tgt_pos]
        best_src_pos = best_src_positions[tgt_pos]

        # stats
        if tgt_token == unk_symbol:
          num_global_unks += 1
        #  if best_src_pos == -1:
        #    num_global_unaligned_unks += 1
        #    num_aligned_unks += 1
        #  else:
        #    num_global_aligned_unks += 1
        #    if tgt_pos>=best_src_pos: # forward movement
        #      forw_dist += tgt_pos-best_src_pos
        #      global_forw_dist += tgt_pos-best_src_pos
        #    else:
        #      back_dist -= tgt_pos-best_src_pos
        #      global_back_dist -= tgt_pos-best_src_pos

        # annotate
        if opt==1:
          if best_src_pos==-1: # try to approximate it
            if tgt_pos>0 and best_src_positions[tgt_pos-1]!=-1: # look left
              best_src_pos = best_src_positions[tgt_pos-1]
              search_count = 1
            
            if tgt_pos<(len(tgt_tokens)-1) and best_src_positions[tgt_pos+1]!=-1: # look right
              if search_count==0:
                best_src_pos = best_src_positions[tgt_pos+1]
              else:
                best_src_pos = best_src_pos + best_src_positions[tgt_pos+1]
              search_count = search_count + 1
            
            if search_count>0: # found an approximation
              best_src_pos = best_src_pos/search_count
              approx_align_found = approx_align_found+1 

            else: # no approximation, use use the tgt_pos
              approx_align_notfound = approx_align_notfound+1 
              best_src_pos = tgt_pos
            
          else:
            num_aligns = num_aligns + 1
            if tgt_pos>=best_src_pos: # forward movement
              global_forw_dist += tgt_pos-best_src_pos
            else:
              global_back_dist -= tgt_pos-best_src_pos

          # make sure the best_src_pos is valid
          if best_src_pos<0:
            best_src_pos=0
          elif best_src_pos>(len(src_unk_tokens)-2): # exclude eos
            best_src_pos = len(src_unk_tokens)-2

          # make sure best_src_pos is within the window
          if is_absolute==0:
            if best_src_pos>(tgt_pos+window):
              best_src_pos = tgt_pos+window
            elif best_src_pos<(tgt_pos-window):
              best_src_pos = tgt_pos-window

          if is_absolute: # absolute
            tgt_unk_positions.append(str(best_src_pos))
          else: # relative
            tgt_unk_positions.append(str(tgt_pos-best_src_pos))

            #if best_src_pos ==-1: # unaligned
            #  pos_word = pos_null
            #  if debug: sys.stderr.write('  null aligned: %s\n' % (tgt_tokens[tgt_pos]))
            #else:
            #  if debug: 
            #    sys.stderr.write('  aligned: %s\t%s\n' % (src_tokens[best_src_pos], tgt_tokens[tgt_pos]))
            #  if best_src_pos < (tgt_pos-window) or best_src_pos > (tgt_pos+window): # out of boundary, consider null
            #    if debug: 
            #      sys.stderr.write('  null aligned (out boundary): %s, best_src_pos=%d\n' % (tgt_tokens[tgt_pos], best_src_pos))
            #    pos_word = pos_null
            #  else:
            #    pos_word = '<p_' + str(tgt_pos-best_src_pos) + '>'
            #tgt_unk_tokens.append(pos_word)

          tgt_unk_tokens.append(tgt_token)
        else: # opt=4
          if tgt_token==unk_symbol:
            if best_src_pos!=-1: # aligned 
              assert best_src_pos >= (tgt_pos-window) and best_src_pos <= (tgt_pos+window)
              tgt_token = '<unk_' + str(tgt_pos-best_src_pos) + '>'
          tgt_unk_tokens.append(tgt_token)
          
    elif opt==2: # annotate copying unks
      tgt_unk_tokens = list(tgt_tokens)
      tgt_pos_flags = {} # keep track of positions we have processed
      tgt_unk_flags = {} # keep track of unique unk symbols we have used
      for tgt_pos in t2s.keys():
        if len(t2s[tgt_pos])>1: continue
        src_pos = t2s[tgt_pos][0] 
        src_token = src_tokens[src_pos]
        tgt_token = tgt_tokens[tgt_pos]

        if src_token not in src_vocab_map and tgt_token not in tgt_vocab_map: # both unk  
          # check if the two words are possible translations of each other
          if (dict_size>0 and src_token in dict_map and tgt_token in dict_map[src_token] and dict_map[src_token][tgt_token]>0.5) or (src_token == tgt_token):
            unk = src_unk_tokens[src_pos]
            tgt_unk_tokens[tgt_pos] = unk # copy over
            
            tgt_pos_flags[tgt_pos] = 1
            tgt_unk_flags[unk] = 1
            num_aligned_unks += 1 # aligned unks

      # now go one more time through the target side and annotate the rest
      for tgt_pos in xrange(len(tgt_tokens)):
        if tgt_pos in tgt_pos_flags: # already copy unks
          continue

        tgt_token = tgt_tokens[tgt_pos]
        if tgt_token not in tgt_vocab_map: # rare word
          unk = unk_symbol
          tgt_unk_tokens[tgt_pos] = unk

    # write text
    if src_output_opt == 1 or src_output_opt == 2 or src_output_opt == 3: # manipulate src
      src_len = len(src_unk_tokens)
      reverse_src_unk_tokens = [src_unk_tokens[src_len-2-ii] for ii in xrange(src_len-1)] # ignore eos
      reverse_src_unk_tokens.append(eos)
      if src_output_opt == 1: # reverse
        src_unk_tokens = reverse_src_unk_tokens
      elif src_output_opt == 2: # forward src, reverse src
        src_unk_tokens[-1] = src_split # replace eos by src_split
        src_unk_tokens.extend(reverse_src_unk_tokens)
      elif src_output_opt == 3: # reverse src, reverse src
        src_unk_tokens = list(reverse_src_unk_tokens)
        src_unk_tokens[-1] = src_split # replace eos by src_split
        src_unk_tokens.extend(reverse_src_unk_tokens)
        
    if no_eos:
      src_ouf.write('%s\n' %  (' '.join(src_unk_tokens[0:-1])))
    else:
      src_ouf.write('%s\n' %  (' '.join(src_unk_tokens)))

    # convert to integers
    if is_separate_output: # for separate output, we don't increase indices by tgt_vocab_size
      src_indices = text.to_id(src_unk_tokens, src_vocab_map)
    else:
      src_indices = text.to_id(src_unk_tokens, src_vocab_map, tgt_vocab_size)

    if opt==1:
      new_tgt_tokens = []
      # combine positions and tgt word indices
      for ii in xrange(len(tgt_unk_tokens)):
        new_tgt_tokens.append(tgt_unk_positions[ii])
        new_tgt_tokens.append(tgt_unk_tokens[ii])
      tgt_ouf.write('%s\n' %  (' '.join(new_tgt_tokens))) # this one doesn't have <eos> at the end
    else:  
      tgt_ouf.write('%s\n' %  (' '.join(tgt_unk_tokens))) # this one doesn't have <eos> at the end

    tgt_unk_tokens.append(eos)
    tgt_indices = text.to_id(tgt_unk_tokens, tgt_vocab_map)
    if opt==1:
      new_tgt_indices = []
      # combine positions and tgt word indices
      for ii in xrange(len(tgt_indices)-1):
        new_tgt_indices.append(tgt_unk_positions[ii])
        new_tgt_indices.append(tgt_indices[ii])
      new_tgt_indices.append(tgt_indices[-1])
      tgt_indices = new_tgt_indices

    if is_separate_output:
      if no_eos:
        src_id_ouf.write('%s\n' %  (' '.join(src_indices[0:-1]))) 
        tgt_id_ouf.write('%s\n' %  (' '.join(tgt_indices[0:-1]))) 
      else:
        src_id_ouf.write('%s\n' %  (' '.join(src_indices))) 
        tgt_id_ouf.write('%s\n' %  (' '.join(tgt_indices))) 
    else:
      if no_eos:
        src_tgt_id_ouf.write('%s %s %s\n' %  (' '.join(src_indices[0:-1]), delim, ' '.join(tgt_indices[0:-1]))) 
      else:
        src_tgt_id_ouf.write('%s %s %s\n' %  (' '.join(src_indices), delim, ' '.join(tgt_indices))) 
    
    # debug
    if debug == True and num_aligns>0: #num_aligned_unks>0 and forw_dist>0 and back_dist>0:
      print('\n--- Debug 1st sent ---')
      print('src orig: ' + src_line) 
      print('tgt orig: ' + tgt_line) 
      print('align   : ' + align_line) 
      print('\nsrc  unk: ' + ' '.join(src_unk_tokens))
      print('tgt  unk: ' + ' '.join(tgt_unk_tokens))
      if opt==2:
        print('\naligned  unk: ' + ' '.join([tgt_unk_tokens[i] for i in tgt_pos_flags.keys()]))
      print('\nsrc   id: ' + ' '.join(src_indices))
      print('tgt   id: ' + ' '.join(tgt_indices))
      if is_separate_output:
        print('\nsrc  unk: ' + ' '.join(text.to_text(src_indices, src_words)))
      else:
        print('\nsrc  unk: ' + ' '.join(text.to_text(src_indices, src_words, tgt_vocab_size)))
      print('tgt  unk: ' + ' '.join(text.to_text(tgt_indices, tgt_words)))
      debug = False
      print('--- End debug 1st sent ---\n')
    line_id = line_id + 1
    if (line_id % 10000 == 0):
      sys.stderr.write(' (%d) ' % line_id)

  sys.stderr.write('  num lines = %d, unk=%.2f, forw dist=%.2f, back dist=%.2f, num_aligns=%d, approx_align_found=%d, approx_align_notfound=%d\n' % (line_id, float(num_global_unks)/line_id, float(global_forw_dist)/line_id, float(global_back_dist)/line_id, num_aligns, approx_align_found, approx_align_notfound))
  #sys.stderr.write('  num lines = %d, unk=%.2f, aligned unk=%.2f, unaligned unk=%.2f, forw dist=%.2f, back dist=%.2f, num_aligns=%d, approx_align_found=%d, approx_align_notfound=%d\n' % (line_id, float(num_global_unks)/line_id, float(num_global_aligned_unks)/line_id, float(num_global_unaligned_unks)/line_id, float(global_forw_dist)/line_id, float(global_back_dist)/line_id, num_aligns, approx_align_found, approx_align_notfound))
  src_inf.close()
  tgt_inf.close()
  if opt>0:
    align_inf.close()

  if is_separate_output:
    src_id_ouf.close()
    tgt_id_ouf.close()
  else:
    src_tgt_id_ouf.close()
  src_ouf.close()
  tgt_ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_prefix, args.src_lang, args.tgt_lang, args.out_prefix, args.freq, args.is_reverse_alignment, args.opt, args.dict_file, args.src_vocab_file, args.tgt_vocab_file, args.src_vocab_size, args.tgt_vocab_size, args.src_output_opt, args.is_separate_output, args.is_separate_pos, args.no_eos, args.window, args.is_absolute)


            #search_count = 0 
            #if tgt_pos>0 and best_src_positions[tgt_pos-1]!=-1:
            #  best_src_pos = best_src_positions[tgt_pos-1]
            #  search_count = 1
            #
            #if tgt_pos<(len(tgt_tokens)-1) and best_src_positions[tgt_pos+1]!=-1:
            #  if search_count==0:
            #    best_src_pos = best_src_positions[tgt_pos+1]
            #  else:
            #    best_src_pos = best_src_pos + best_src_positions[tgt_pos+1]
            #  search_count = search_count + 1
            #
            #if search_count>0: # found an approximation
            #  best_src_pos = best_src_pos/search_count

            #  if best_src_pos>(tgt_pos+window):
            #    best_src_pos = tgt_pos+window
            #  elif best_src_pos<(tgt_pos-window):
            #    best_src_pos = tgt_pos-window
            #else: # use use the tgt_pos
            #  best_src_pos = tgt_pos
            #
            ## make sure the src pos is valid
            #if best_src_pos<0:
            #  best_src_pos=0
            #elif best_src_pos>(len(src_unk_tokens)-2): # exclude eos
            #  best_src_pos = len(src_unk_tokens)-2

