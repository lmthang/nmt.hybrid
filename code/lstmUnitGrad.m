function [dc, dh, lstm_grad] = lstmUnitGrad(model, lstm, x_t, dc, dh, t, srcMaxLen, zero_state, params)
  dc = dc + params.nonlinear_f_prime(lstm{t}.f_c_t).*lstm{t}.o_gate.*dh;

  di = params.nonlinear_gate_f_prime(lstm{t}.i_gate).*lstm{t}.a_signal.*dc;
  if t>1
    df = params.nonlinear_gate_f_prime(lstm{t}.f_gate).*lstm{t-1}.c_t.*dc;
  else
    df = zero_state;
%     if params.isGPU
%       df = zeros(params.lstmSize, curBatchSize, dataType, 'gpuArray');
%     else
%       df = zeros(params.lstmSize, curBatchSize);
%     end
  end

  % arrayfun doesn't work here for GPU
  do = params.nonlinear_gate_f_prime(lstm{t}.o_gate).*lstm{t}.f_c_t .* dh;
  da = params.nonlinear_f_prime(lstm{t}.a_signal).*lstm{t}.i_gate.*dc;   

  if (t>=srcMaxLen) % grad tgt
    W = model.W_tgt;
    if t==1
      lstm_grad.W_tgt = [di; df; do; da]*[x_t; zero_state]';
    else
      lstm_grad.W_tgt = [di; df; do; da]*[x_t; lstm{t-1}.h_t]';
    end
  else % grad src
    W = model.W_src;
    if t==1
      lstm_grad.W_src = [di; df; do; da]*[x_t; zero_state]';
    else
      lstm_grad.W_src = [di; df; do; da]*[x_t; lstm{t-1}.h_t]';
    end
  end

  lstm_grad.dx = W(:, 1:params.lstmSize)'*[di; df; do; da];
  dc = lstm{t}.f_gate.*dc;
  dh = W(:, params.lstmSize+1:end)'*[di; df; do; da];
  
  % clip hidden/cell derivatives
  if params.isClip
    if params.isGPU
     dh = arrayfun(@clipBackward, dh);
     dc = arrayfun(@clipBackward, dc);
    else
     dh(dh>params.clipBackward) = params.clipBackward; dh(dh<-params.clipBackward) = -params.clipBackward;
     dc(dc>params.clipBackward) = params.clipBackward; dc(dc<-params.clipBackward) = -params.clipBackward;
    end
  end
end

function [clippedValue] = clipBackward(x)
  if x>1000
    clippedValue = single(1000);
  elseif x<-1000
    clippedValue = single(-1000);
  else
    clippedValue = x;
  end
end