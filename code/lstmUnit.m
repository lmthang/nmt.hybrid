%% LSTM unit
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
% Input:
%   W: parameter
%   x_t: current input
%   h_t, c_t: previous hidden state, cell state
%
% Output:
%   lstm struct
%%
function [lstmCell] = lstmUnit(W, x_t, h_t, c_t, params)
  %% input, forget, output gates and input signals before applying non-linear functions
  ifoa_linear = W*[x_t; h_t];    

  %% cell
  % GPU note: the below non-linear functions are fast, so no need to use arrayfun
  ifo_gate = params.nonlinear_gate_f(ifoa_linear(1:3*params.lstmSize, :));
  lstmCell.i_gate = ifo_gate(1:params.lstmSize, :);
  lstmCell.f_gate = ifo_gate(params.lstmSize+1:2*params.lstmSize, :);
  lstmCell.o_gate = ifo_gate(2*params.lstmSize+1:3*params.lstmSize, :);
  lstmCell.a_signal = params.nonlinear_f(ifoa_linear(3*params.lstmSize+1:4*params.lstmSize, :)); % note input uses a different activation function
  lstmCell.c_t = lstmCell.f_gate.*c_t + lstmCell.i_gate.*lstmCell.a_signal; % c_t = f_t * c_{t-1} + i_t * a_t

  %% hidden
  lstmCell.f_c_t = params.nonlinear_f(lstmCell.c_t);
  lstmCell.h_t = lstmCell.o_gate.*lstmCell.f_c_t; % h_t = o_t * g(c_t)

  %% clip
  if params.isClip
    if params.isGPU
     lstmCell.c_t = arrayfun(@clipForward, lstmCell.c_t);
     lstmCell.h_t = arrayfun(@clipForward, lstmCell.h_t);
    else
     lstmCell.c_t(lstmCell.c_t>params.clipForward) = params.clipForward; lstmCell.c_t(lstmCell.c_t<-params.clipForward) = -params.clipForward; % clip: keep memory small
     lstmCell.h_t(lstmCell.h_t>params.clipForward) = params.clipForward; lstmCell.h_t(lstmCell.h_t<-params.clipForward) = -params.clipForward; % clip: keep hidden state small
    end
  end
end


function [clippedValue] = clipForward(x)
  if x>50
    clippedValue = single(50);
  elseif x<-50
    clippedValue = single(-50);
  else
    clippedValue = x;
  end
end
