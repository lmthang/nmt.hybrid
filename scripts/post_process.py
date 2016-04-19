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
  parser.add_argument('tgt_file', metavar='tgt_file', type=str, help='translation file') 
  parser.add_argument('align_file', metavar='align_file', type=str, help='align file') 
  parser.add_argument('dict_file', metavar='dict_file', type=str, help='dict file') 
  parser.add_argument('ref_file', metavar='ref_file', type=str, help='ref file') 
  parser.add_argument('out_file', metavar='out_file', type=str, help='output file') 

  # optional arguments
  parser.add_argument('-s', '--src_sgm', dest='src_sgm', type=str, default='', help='src file in SGM format to compute NIST BLEU score with mteval-v13a.pl')
  parser.add_argument('-t', '--tgt_sgm', dest='tgt_sgm', type=str, default='', help='tgt file in SGM format to compute NIST BLEU score with mteval-v13a.pl')
  parser.add_argument('-l', '--lang', dest='lang', type=str, default='', help='tgt lang, e.g., de, en, etc., to be used with mteval-v13a.pl')
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
  sys.stderr.write('# Loading dict file %s\n' % dict_file)
  line_id = 0
  dict_map = {}
  prob_map = {}
  for line in inf:
    tokens = re.split('\s+', line.strip())
    if len(tokens) != 3:
      sys.stderr.write('# Skip line: %s' % line)
      continue
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

def nist_bleu(script_dir, trans_file, src_sgm, tgt_sgm, lang):
  detok_trans_file = trans_file + '.detok'

  tok_lang = lang
  if tok_lang == 'german': # adhoc for IWSLT
    tok_lang = 'de'
  cmd = 'perl %s/wmt/detokenizer.pl -l %s < %s > %s' % (script_dir, lang, trans_file, detok_trans_file)
  os.system(cmd)
  trans_file = detok_trans_file
  
  trans_sgm = trans_file + '.sgm'
  cmd = '%s/wmt/wrap-xml.perl %s %s ours < %s > %s' % (script_dir, lang, src_sgm, trans_file, trans_sgm)
  os.system(cmd)

  cmd = 'perl %s/wmt/mteval-v13a.pl -d 0 -r %s -s %s -t %s -c' % (script_dir, tgt_sgm, src_sgm, trans_sgm)
  os.system(cmd)

def bleu(script_dir, trans_file, ref_file):
  cmd = script_dir + '/wmt/multi-bleu.perl ' + ref_file + ' < ' + trans_file
  sys.stderr.write('# BLEU: %s\n' % cmd)
  os.system(cmd)

def chr_f(script_dir, trans_file, ref_file):
  cmd = script_dir + '/chrF.py --ref ' + ref_file + ' --hyp ' + trans_file
  sys.stderr.write('# 6-gram chrF3: %s\n' % cmd)
  os.system(cmd)

def process_files(align_file, src_file, tgt_file, ref_file, dict_file, out_file, src_sgm, tgt_sgm, lang, is_reverse_alignment):
  """
  """
  tgt_inf = codecs.open(tgt_file, 'r', 'utf-8')

  is_src = 0
  if src_file != '':
    is_src = 1
    src_inf = codecs.open(src_file, 'r', 'utf-8')

  is_align = 0
  if align_file != '':
    is_align = 1
    align_inf = codecs.open(align_file, 'r', 'utf-8')

  is_ref = 0
  if ref_file != '':
    ref_inf = codecs.open(ref_file, 'r', 'utf-8')
    is_ref = 1

  # load dict
  is_dict = 0
  if dict_file != '' and os.path.exists(dict_file):
    dict_map = load_dict(dict_file)
    is_dict = 1

  # out_file
  if out_file == '':
    out_file = tgt_file + '.post'
  ouf = codecs.open(out_file, 'w', 'utf-8')

  new_tgt_file = tgt_file + '.new'
  new_tgt_ouf = codecs.open(new_tgt_file, 'w', 'utf-8')

  # post process
  unk = '<unk>'
  line_id = 0
  debug = 1
  unk_count = 0
  dictionary_count = 0
  identity_count = 0
  for tgt_line in tgt_inf:
    tgt_line = tgt_line.strip()
    debug_count = 0
    debug_str = ''

    if is_src:
      src_line = src_inf.readline().strip()
    if is_ref:
      ref_line = ref_inf.readline().strip()
    if is_align:
      src_tokens = re.split('\s+', src_line)
      tgt_tokens = re.split('\s+', tgt_line)

      # get alignment
      align_line = align_inf.readline().strip()
      if is_reverse_alignment==True: # reversed alignment tgtId-srcId
        (t2s, s2t) = text.aggregate_alignments(align_line)
      else: # normal alignment srcId-tgtId
        (s2t, t2s) = text.aggregate_alignments(align_line)
       
      new_tgt_tokens = []
      for tgt_pos in xrange(len(tgt_tokens)):
        tgt_token = tgt_tokens[tgt_pos]
        if tgt_tokens[tgt_pos] == unk and is_dict:
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

        #if tgt_token != '##AT##-##AT##':
        new_tgt_tokens.append(tgt_token)

      out_line = ' '.join(new_tgt_tokens)
    else:
      out_line = tgt_line

    # post process
    if re.search('##AT##-##AT##', out_line):
      out_line = re.sub(' ##AT##-##AT## ', '-', out_line)
      tgt_line = re.sub(' ##AT##-##AT## ', '-', tgt_line)
      if is_align == 0:
        debug_count = 1
    ouf.write('%s\n' % out_line)
    new_tgt_ouf.write('%s\n' % tgt_line)

    # debug info
    if debug == 1 and debug_count>0:
      sys.stderr.write('# example %d\n' % line_id)
      if is_src:
        sys.stderr.write('src: %s\n' % (src_line))
      sys.stderr.write('tgt: %s\n' % (tgt_line))
      sys.stderr.write('%s' % (debug_str))
      sys.stderr.write('out: %s\n' % (out_line))
      if is_ref:
        sys.stderr.write('ref: %s\n' % ref_line)
      debug = 0

    line_id += 1   # concat results

  if is_src:
    src_inf.close()
  if is_align:
    align_inf.close()
  tgt_inf.close()
  ouf.close()
  new_tgt_ouf.close()
  sys.stderr.write('# num sents = %d, unk count=%d, dictionary_count=%d, identity_count=%d\n' % (line_id, unk_count, dictionary_count, identity_count))

  # evaluating 
  if is_ref:
    script_dir = os.path.dirname(sys.argv[0])
    bleu(script_dir, new_tgt_file, ref_file)
   
    if is_align:
      chr_f(script_dir, out_file, ref_file)
      bleu(script_dir, out_file, ref_file)
      if src_sgm != '' and tgt_sgm != '' and lang != '': # compute NIST BLEU score
        nist_bleu(script_dir, out_file, src_sgm, tgt_sgm, lang)


if __name__ == '__main__':
  args = process_command_line()
  process_files(args.align_file, args.src_file, args.tgt_file, args.ref_file, args.dict_file, args.out_file, args.src_sgm, args.tgt_sgm, args.lang, args.is_reverse_alignment)
