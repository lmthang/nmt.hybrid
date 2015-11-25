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
  parser.add_argument('out_file', metavar='out_file', type=str, help='output directory') 

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

def process_files(in_file, out_file):
  """
  Read data from in_file, and output to out_file
  """

  sys.stderr.write('# in_file = %s, out_file = %s\n' % (in_file, out_file))
  # input
  sys.stderr.write('# Input from %s.\n' % (in_file))
  inf = open(in_file, 'r')

  # output
  sys.stderr.write('Output to %s\n' % out_file)
  check_dir(out_file)
  ouf = codecs.open(out_file, 'w', 'utf-8')

  title = 'Learning curve'
  xLabel = 'Mini-batches'
  yLabel = 'Test cost'
  ouf.write('%s\n%s\n%s' % (title, yLabel, xLabel))

  line_id = 0
  sys.stderr.write('# Processing file %s ...\n' % (in_file))
   
  eval_pattern = re.compile('# eval (.+), train=([\d\.]+).*, valid=([\d\.]+).*, test=([\d\.]+).*,')
  models = []
  train_stats = {}
  test_stats = {}
  model_count = 0
  log_freqs = []
  for line in inf:
    tokens = line.split()
    file_name = tokens[0]
    model_count = model_count + 1
    if len(tokens)>1:
      model = tokens[1]
    else:
      model = 'model' + str(model_count)
    # log
    log_file = os.path.expanduser(file_name + '/log')
    if os.path.exists(log_file):
      log_inf = codecs.open(log_file, 'r', 'utf-8')
      models.append(model)
      train_stats[model] = {}
      sys.stderr.write('# model %s\n%s\n' % (model, log_file))
      ouf.write('\t%s' % model)
     
      prev_iter = -1
      done_log_freq = False
      for line in log_inf:
        eval_m = re.search(eval_pattern, line)
        if eval_m != None:
          eval_stat = eval_m.group(1)
          print eval_stat       
          tokens = re.split(', ', eval_stat)
          cur_iter = int(tokens[-3])
          train_cost = eval_m.group(2)
          test_cost = eval_m.group(4)
          print cur_iter, train_cost, test_cost
          #if cur_iter not in train_stats[model]:
          train_stats[model][cur_iter] = test_cost
          
          # guess log freq
          if prev_iter == -1:
            prev_iter = cur_iter
          elif done_log_freq == False and prev_iter > -1:
            done_log_freq = True
            log_freqs.append(cur_iter - prev_iter)
      log_inf.close()
  ouf.write('\n')

  num_models = len(models) 
  sys.stderr.write('# Num models = %d\n' % num_models)
  #log_freq = 5000

  # make sure all models have the same log freq
  for i in xrange(1, num_models):
    assert log_freqs[i] == log_freqs[0]
  log_freq = log_freqs[0]
  sys.stderr.write('# log_freq = %d\n' % log_freq)

  cur_iter = 0
  while(1):
    cur_iter += log_freq
    results = []
    for ii in xrange(num_models):
      model = models[ii]
      if cur_iter in train_stats[model]:
        results.append(train_stats[model][cur_iter])
    if len(results)==0:
      break
    if len(results)==num_models: # have train_stats for all models
      ouf.write('%d\t%s\n' % (cur_iter, '\t'.join(results)))
    else:
      sys.stderr.write('iter %d, only %d models\n' % (cur_iter, len(results)))
    
  inf.close()
  ouf.close()

  ## save image
  #os.chdir('/home/lmthang/bin/misc')
  #command = '/afs/cs/software/bin/matlab_r2014b -nodesktop -nodisplay -nosplash -r \"plotData(\'%s\',1,1,1,\'outFile\',\'%s.png\');exit;\"' % (out_file, out_file)
  #sys.stderr.write('# Executing: %s\n' % command)
  #os.system(command)

if __name__ == '__main__':
  args = process_command_line()
  process_files(args.in_file, args.out_file)


