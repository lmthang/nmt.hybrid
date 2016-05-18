#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# Script dir = $DIR"
echo "cd $DIR/../code"
cd $DIR/../code
commands_str="trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'attnFunc', 0, 'feedInput', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'attnFunc', 0, 'feedInput', 1, 'numLayers', 2);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'attnFunc', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 1, 'attnOpt', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 2, 'attnOpt', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 4, 'attnOpt', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 1, 'attnOpt', 2);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 2, 'attnOpt', 2);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 4, 'attnOpt', 2);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 4, 'attnOpt', 2, 'normLocalAttn', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 1, 'attnOpt', 3);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 2, 'attnOpt', 3);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 4, 'attnOpt', 3);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'attnFunc', 0, 'feedInput', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'attnFunc', 0, 'feedInput', 1, 'numLayers', 2);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'attnFunc', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 1, 'attnOpt', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 2, 'attnOpt', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 4, 'attnOpt', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 1, 'attnOpt', 2);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 2, 'attnOpt', 2);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 4, 'attnOpt', 2);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 4, 'attnOpt', 2, 'normLocalAttn', 1);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 1, 'attnOpt', 3);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 2, 'attnOpt', 3);\
trainLSTM('', '', '', '', '', '', '', '../output/gradcheck', 'isGradCheck', 1, 'initRange', 10, 'isResume', 0, 'feedInput', 1, 'numLayers', 2, 'dropout', 0.8, 'isReverse', 1, 'attnFunc', 4, 'attnOpt', 3)"

IFS=';' read -a commands <<< "$commands_str"
for command in "${commands[@]}"
do
  echo ""
  echo "## $command"
  #echo "matlab -nodesktop -nodisplay -nosplash -r \"try; $matlabCommand ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()\""
  matlab -nodesktop -nodisplay -nosplash -r "try; $command ; catch ME; fprintf('\n! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message); for k=1:length(ME.stack); fprintf('stack %d: file=%s, name=%s, line=%d\n', k, ME.stack(k).file, ME.stack(k).name, ME.stack(k).line); end; end; exit()"
done
