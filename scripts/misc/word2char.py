#!/usr/bin/env python

"""
  Convert words to char ngrams.
"""


__author__ = 'Thang Luong'

import sys
import re
import os
import codecs

reload(sys)
sys.setdefaultencoding('utf-8')


def deescape(line):
  """ de-escape special chars """
  line = re.sub('&bar;', '|', line)   # factor separator (legacy)
  line = re.sub('&#124;', '|', line)  # factor separator
  line = re.sub('&lt;', '<', line)    # xml
  line = re.sub('&gt;', '>', line)    # xml
  line = re.sub('&bra;', '[', line)   # syntax non-terminal (legacy)
  line = re.sub('&ket;', ']', line)   # syntax non-terminal (legacy)
  line = re.sub('&quot;', '\"', line)  # xml
  line = re.sub('&apos;', '\'', line)  # xml
  line = re.sub('&#91;', '[', line)   # syntax non-terminal
  line = re.sub('&#93;', ']', line)   # syntax non-terminal
  line = re.sub('&amp;', '&', line)   # escape escape
  return line

def main(in_file, out_file, ngram_size):
  """Main body"""

  print "# ngram_size = ", ngram_size
  inf = codecs.open(in_file, 'r', 'utf-8')
  ouf = codecs.open(out_file, 'w', 'utf-8')

  # get to the correct dir
  sys.stderr.write('# Cur dir: %s\n' % os.getcwd()) 

  # read input file
  line_count = 0
  word_count = 0
  ngram_count = 0
  word_boundary = "#B#"
  for line in inf:
    line = deescape(line.strip())
    tokens = line.split()
    ngrams = []
    num_tokens = len(tokens)

    # split to ngrams
    for ii in xrange(num_tokens):
      word = tokens[ii]
      num_chars = len(word)
      start_id = 0
      while start_id < num_chars:
        end_id = start_id+ngram_size
        if end_id > num_chars:
          end_id = num_chars
        ngram = word[start_id:end_id]

        # if start_id == 0: # begin of word
        #   ngram = '#B#' + ngram
        #if end_id == num_chars: # end of word
        #  ngram += '#E#'

        ngrams.append(ngram)
        start_id += ngram_size
      ngrams.append(word_boundary)

    ouf.write('%s\n' % ' '.join(ngrams))
    if line_count == 0:
      sys.stderr.write('%s -> %s\n' % (line, ' '.join(ngrams)))

    # stats
    line_count += 1
    word_count += num_tokens
    ngram_count += len(ngrams)
    if line_count % 100000 == 0:
      sys.stderr.write('(%dK) ' % int(line_count/1000))

  sys.stderr.write('\n# Total %d lines, %d words, %d ngrams, avg len = %g, avg words = %g\n' % (line_count, word_count, ngram_count, float(ngram_count/line_count), float(word_count/line_count)))
  inf.close()
  ouf.close()

if __name__ == '__main__':
  ngram_size = int(sys.argv[1])
  in_file = sys.argv[2]
  out_file = sys.argv[3]
  main(in_file, out_file, ngram_size)
