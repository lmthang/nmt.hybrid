#!/usr/bin/env python
# Author: Thang Luong <luong.m.thang@gmail.com>, created on Tue Mar  3 14:41:27 PST 2015

"""
Module docstrings.
"""

usage = 'USAGE DESCRIPTION.' 

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
  parser.add_argument('in_file', metavar='in_file', type=str, help='list of log files') 
  parser.add_argument('out_dir', metavar='out_dir', type=str, help='output directory') 

  # optional arguments
  parser.add_argument('-o', '--option', dest='opt', type=int, default=0, help='option (default=0)')
  
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

def process_files(in_file, out_dir):
  """
  Read data from in_file, and output to out_dir
  """

  sys.stderr.write('# in_file = %s, out_dir = %s\n' % (in_file, out_dir))
  # input
  sys.stderr.write('# Input from %s.\n' % (in_file))
  inf = open(in_file, 'r')

  # output
  sys.stderr.write('Output to %s\n' % out_dir)
  #check_dir(out_dir)
  #ouf = codecs.open(out_file, 'w', 'utf-8')

  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
  pattern = re.compile('save model test perplexity ([\d\.]+) ')
  eval_pattern = re.compile('# eval (.+), train')
  err_pattern = re.compile('(JOB \d+ CANCELLED AT .+)')
  save_pattern = re.compile('save model test perplexity ')
  progress_pattern = re.compile('gN=.+?,\s+(.+)')
  results = []
  for file_name in inf:
    file_name = clean_line(file_name)
    if file_name == '':
      results.append('')
      sys.stderr.write('\n')
      continue
   
    # log
    log_file = os.path.expanduser(file_name + '/log')
    best_ppl = ''
    eval_stat = ''
    prev_line = ''
    if os.path.exists(log_file):
      log_inf = codecs.open(log_file, 'r', 'utf-8')
      for line in log_inf:
        save_m = re.search(save_pattern, line)
        if save_m != None:
          eval_m = re.search(eval_pattern, prev_line)
          if eval_m != None:
            eval_stat = eval_m.group(1)

        prev_line = line
      log_inf.close()

    # stderr
    stderr_file = os.path.expanduser(file_name + '/stderr')
    if not os.path.exists(stderr_file):
      stderr_file = os.path.expanduser(file_name + '.stderr')
    err_stat = ''
    status = 'training'
    latest_time = ''
    if os.path.exists(stderr_file):
      stderr_inf = codecs.open(stderr_file, 'r', 'utf-8')
      for line in stderr_inf: 
        m = re.search(err_pattern, line)
        if m != None:
          err_stat = m.group(1)
        if re.search('Done training', line):
          status = 'done'
        m = re.search(progress_pattern, line)
        if m != None:
          latest_time = m.group(1)
      stderr_inf.close()
    eval_stat = status + ' ' + eval_stat
    
    results.append(eval_stat)
    sys.stderr.write('%s %s %s %s\n' % (eval_stat, latest_time, file_name, err_stat))
  
  sys.stderr.write('%s\n' % '\n'.join(results))
  inf.close()
  #ouf.close()

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.out_dir)


