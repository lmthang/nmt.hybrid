"""
"""
import os
import sys
import re
import codecs

def aggregate_alignments(align_line):
  align_tokens = re.split('\s+', align_line.strip())
  s2t = {}
  t2s = {}
  # process alignments
  for align_token in align_tokens:
    if align_token=='': continue
    (src_pos, tgt_pos) = re.split('\-', align_token)
    src_pos = int(src_pos)
    tgt_pos = int(tgt_pos)
    if src_pos not in s2t: s2t[src_pos] = []
    s2t[src_pos].append(tgt_pos)

    if tgt_pos not in t2s: t2s[tgt_pos] = []
    t2s[tgt_pos].append(src_pos)
  
  return (s2t, t2s)

def get_vocab(corpus_file, vocab_file, freq, vocab_size, unk='<unk>'):
  if os.path.isfile(vocab_file): # vocab_file exist
    (words, vocab_map, vocab_size) = load_vocab(vocab_file)
  else:
    (words, vocab_map, freq_map, vocab_size, num_train_words) = load_vocab_from_corpus(corpus_file, freq, vocab_size, unk)
    write_vocab(vocab_file, words, freq_map)

  return (words, vocab_map, vocab_size)


def add_word_to_vocab(word, words, vocab_map, vocab_size):
  if word not in vocab_map:
    words.append(word)
    vocab_map[word] = vocab_size
    vocab_size += 1
    #sys.stderr.write('  add %s\n' % word)
  return (words, vocab_map, vocab_size)

def annotate_unk(tokens, vocab_map, max_num_unks=-1, default_unk='<unk>'):
  sent_unk_map = {} # make sure same words are mapped to the same unk in each sent
  num_unk_types=0
  num_unk_tokens = 0
  unk_tokens = list(tokens)
  for pos in xrange(len(tokens)):
    token = tokens[pos]
       
    if token not in vocab_map:
      if token in sent_unk_map:
        unk = sent_unk_map[token]
      else: # generate a new unk
        if max_num_unks <= 0: # use default unk
          unk = default_unk 
        else:
          if num_unk_types>=max_num_unks: # this sent has lots of unk, use <unk>
            unk = default_unk 
          else:
            unk = '<unk' + str(num_unk_types) + '>'
            num_unk_types += 1

        sent_unk_map[token] = unk

      unk_tokens[pos] = unk
      num_unk_tokens += 1

  return (unk_tokens, num_unk_tokens, num_unk_types)

def to_id(tokens, vocab_map, offset=0, unk='<unk>'):
  return [str(vocab_map[token]+offset) if token in vocab_map else str(vocab_map[unk]+offset) for token in tokens]
    
def to_text(indices, words, offset=0):
  return [words[int(index)-offset] for index in indices]
  
def write_vocab(out_file, words, freq_map):
  f = codecs.open(out_file, 'w', 'utf-8')
  sys.stderr.write('# Output vocab to %s ...\n' % out_file)
  vocab_size = 0
  for word in words:
    #f.write('%s %d\n' % (word, vocab_size))
    f.write('%s\n' % word)
    vocab_size += 1
  f.close()
  sys.stderr.write('  num words = %d\n' % vocab_size)

  # output sorted vocab
  #out_file += '.sorted'
  #f = codecs.open(out_file, 'w', 'utf-8')
  #sys.stderr.write('# Output sorted vocab to %s ...\n' % out_file)
  #vocab_size = 0
  #sorted_vocab = sorted(freq_map.iteritems(), key=lambda x: x[1], reverse=True)
  #for (word, count) in sorted_vocab:
  #  f.write('%s %d\n' % (word, count))
  #  vocab_size += 1
  #f.close()
  #sys.stderr.write('  num words = %d\n' % vocab_size)


def load_dict(in_file):
  sys.stderr.write('# Loading dict file %s ...\n' % in_file) 
  dict_inf = codecs.open(in_file, 'r', 'utf-8')
  dict_map = {}
  dict_size = 0
  for line in dict_inf:
    tokens = re.split('\s+', line.strip())
    src_word = tokens[0]
    tgt_word = tokens[1]
    prob = float(tokens[2])

    # add to dict
    if src_word not in dict_map:
      dict_map[src_word] = {}
    dict_map[src_word][tgt_word] = prob
    dict_size += 1

    if dict_size % 100000 == 0:
      sys.stderr.write(' (%d) ' % dict_size)
      #if dict_size == 100000: break 
  dict_inf.close()
  sys.stderr.write('Done! dict_size = %d\n' % dict_size)
  return (dict_map, dict_size)

def load_vocab(in_file, sos='<s>', eos='</s>', unk='<unk>'):
  sys.stderr.write('# Loading vocab file %s ...\n' % in_file) 
  vocab_inf = codecs.open(in_file, 'r', 'utf-8')
  words = []
  vocab_map = {}
  vocab_size = 0
  for line in vocab_inf:
    tokens = re.split('\s+', line.strip())
    word = tokens[0]
    words.append(word)
    vocab_map[word] = vocab_size
    vocab_size += 1
  
  # add sos, eos, unk 
  for word in [sos, eos, unk]:
    (words, vocab_map, vocab_size) = add_word_to_vocab(word, words, vocab_map, vocab_size)
  vocab_inf.close()
  sys.stderr.write('  num words = %d\n' % vocab_size)
  return (words, vocab_map, vocab_size)

def load_vocab_from_corpus(in_file, freq, max_vocab_size, unk='<unk>'):
  f = codecs.open(in_file, 'r', 'utf-8')
  sys.stderr.write('# Loading vocab from %s ... ' % in_file)
  
  words = []
  vocab_map = {}
  freq_map = {}
  vocab_size = 0
  if freq == -1 and max_vocab_size == -1:
    max_vocab_size = 10000000
    sys.stderr.write('  change max_vocab_size from -1 to %d\n' % max_vocab_size)
  #else:
  #  sos='<s>'
  #  eos='</s>'
  #  unk='<unk>'
  #  words = [unk, sos, eos]
  #  vocab_map = {unk:0, sos:1, eos:2}
  #  freq_map = {unk:0, sos:0, eos:0}
  #  vocab_size = 3

  num_train_words = 0
  num_lines = 0 
  for line in f:
    tokens = re.split('\s+', line.strip())
    num_train_words += len(tokens)
    for token in tokens:
      if token not in vocab_map:
        words.append(token)
        vocab_map[token] = vocab_size
        freq_map[token] = 0
        vocab_size += 1
      freq_map[token] += 1

    num_lines += 1
    if num_lines % 100000 == 0:
      sys.stderr.write(' (%d) ' % num_lines)
  f.close()
  sys.stderr.write('\n  vocab_size=%d, num_train_words=%d, num_lines=%d\n' % (vocab_size, num_train_words, num_lines))

  # sort, update vocabs e.g., add unk, eos, sos
  (words, vocab_map, freq_map, vocab_size) = update_vocab(words, vocab_map, freq_map, freq, max_vocab_size, unk=unk)
  return (words, vocab_map, freq_map, vocab_size, num_train_words)

def update_vocab(words, vocab_map, freq_map, freq, max_vocab_size, sos='<s>', eos='</s>', unk='<unk>'):
  """
  Filter out rare words (<freq) or keep the top vocab_size frequent words
  """
 
  special_words = [unk, sos, eos]
  new_words = special_words
  new_vocab_map = {unk:0, sos:1, eos:2}
  vocab_size = 3
  if unk in words: # already have unk token
    new_freq_map = {unk: freq_map[unk], sos:0, eos:0}
    del freq_map[unk]
    words.remove(unk)
  else:
    new_freq_map = {unk:0, sos:0, eos:0}

  if freq>0:
    for word in words:
      if freq_map[word] < freq and word not in special_words: # rare
        new_freq_map[unk] += freq_map[word]
      else:
        new_words.append(word)
        new_vocab_map[word] = vocab_size
        new_freq_map[word] = freq_map[word]
        vocab_size += 1
    sys.stderr.write('  convert rare words (freq<%d) to %s: new vocab size=%d, unk freq=%d\n' % (freq, unk, vocab_size, new_freq_map[unk]))
  else:
    assert(max_vocab_size>0)
    sorted_items = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
    for (word, freq) in sorted_items:
      new_words.append(word)
      new_vocab_map[word] = vocab_size
      new_freq_map[word] = freq
      vocab_size += 1
      if vocab_size == max_vocab_size:
        break
    sys.stderr.write('  update vocab: new vocab size=%d\n' % (vocab_size))
  return (new_words, new_vocab_map, new_freq_map, vocab_size)
 
