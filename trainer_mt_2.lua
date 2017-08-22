require 'nn'
require 'sys'
require 'torch'

local Trainer = torch.class('Trainer')

-- Perform one epoch of training.
function Trainer:train(train_data, train_data_au, train_labels, train_labels_au, test, test_au, test_label, test_label_au, dev, dev_au, dev_label, dev_label_au, model, criterion, criterion_au, criterion_au2, optim_method, layers, state, params, grads)
  
  model:training()

  local train_size = train_data:size(1)
  local test_size = test:size(1)
  local dev_size = dev:size(1)
  --local ratio = math.floor(test_size / train_size)

  --if ratio > 1 then
    --local shuffle = torch.randperm(ratio)
    --local t = (shuffle[1] - 1) * train_size + 1
    --test = test:narrow(1, t, train_size)
    --test_au = test_au:narrow(1, t, train_size) --narrow(dim,index, size)
    --test_label = test_label:narrow(1, t, train_size)
    --test_label_au = test_label_au:narrow(1, t, train_size)    
    --test_size = test:size(1) 
  --end
  --print('test')
  --print(test_size)
  --print('train')
  --print(train_size)

  local timer = torch.Timer()
  local time = timer:time().real
  local total_err = 0

  local classes = {}
  for i = 1, opt.num_classes do
    table.insert(classes, i)
  end

  local num_classes = {}
  for i = 1, opt.num_classes_au do
    table.insert(num_classes, i)
  end
  
  local confusion = optim.ConfusionMatrix(classes)
  local confusion2 = optim.ConfusionMatrix(num_classes)
  local confusion3 = optim.ConfusionMatrix(num_classes)
  local confusion4 = optim.ConfusionMatrix(num_classes)
  local confusion5 = optim.ConfusionMatrix(num_classes)
  confusion:zero()
  confusion2:zero()
  confusion3:zero()
  confusion4:zero()
  confusion5:zero()

  local config -- for optim
  if opt.optim_method == 'adadelta' then
    config = { rho = 0.95, eps = 1e-6 } 
  elseif opt.optim_method == 'adam' then
    config = {}
  end

  -- shuffle batches
  local num_batches = math.floor(train_size / opt.batch_size)
  local test_num_batches = math.floor(test_size / opt.batch_size)
  local dev_num_batches = math.floor(dev_size / opt.batch_size)

  --local shuffle = torch.randperm(num_batches)
  local shuffle = torch.range(1, num_batches)
  --local test_shuffle = torch.randperm(test_num_batches)
  local test_shuffle = torch.range(1, test_num_batches)
  --local dev_shuffle = torch.randperm(dev_num_batches)
  local dev_shuffle = torch.range(1, dev_num_batches)

  for i = 1, shuffle:size(1) do
    local t = (shuffle[i] - 1) * opt.batch_size + 1
    local batch_size = math.min(opt.batch_size, train_size - t + 1)

    -- data samples and labels, in mini batches.
    local inputs = train_data:narrow(1, t, batch_size) --narrow(dim,index, size)
	  local inputs_au = train_data_au:narrow(1, t, batch_size) --narrow(dim,index, size)
    local targets = train_labels:narrow(1, t, batch_size)
	  local targets_au = train_labels_au:narrow(1, t, batch_size)  

    --print(inputs[10])
    --print(inputs_au[10])
    --print(targets[10])
    --print(targets_au[10])

    if opt.cudnn == 1 then
      inputs = inputs:cuda()
      targets = targets:cuda()
    else
      inputs = inputs:double()
      targets = targets:double()
    end

    -- closure to return err, df/dx
    local func = function(x)
      -- get new parameters
      if x ~= params then
        params:copy(x)
      end
      -- reset gradients
      grads:zero()

      -- forward pass
      local output_all = model:forward({inputs, inputs_au})
	    local outputs = output_all[1]
      local outputs_au = output_all[2] 
	    local outputs_au2 = output_all[3]
      local err = criterion:forward(outputs, targets)
      --local target1 = targets_au:select(2,1)
      --print(targets)
      --print(target1)

	    local err_au = criterion_au:forward(outputs_au, targets_au:select(2,1))
    --print('hehe')
	    local err_au2 = criterion_au2:forward(outputs_au2, targets_au:select(2,2))

      -- track errors and confusion
      total_err = total_err + err * batch_size
      for j = 1, batch_size do
        confusion:add(outputs[j], targets[j])
        confusion2:add(outputs_au[j], targets_au[j]:select(2,1))
        confusion3:add(outputs_au2[j], targets_au[j]:select(2,2))
      end


      -- compute gradients
      local df_do = criterion:backward(outputs, targets)
	    local df_do_au = criterion_au:backward(outputs_au, targets_au:select(2,1))
	    local df_do_au2 = criterion_au2:backward(outputs_au2, targets_au:select(2,2))
      model:backward({inputs,inputs_au}, {df_do,df_do_au,df_do_au2} )

      if opt.model_type == 'static' then
        -- don't update embeddings for static model
        layers.w2v.gradWeight:zero()
        --layers.w2v2.gradWeight:zero()      
      elseif opt.model_type == 'multichannel' then
        -- keep one embedding channel static
        layers.chan1.gradWeight:zero()
      --elseif opt.model_type == 'nonstatic' then
        ---- don't update embeddings for static model
        --local addw2v = layers.w2v.gradWeight+layers.w2v2.gradWeight
        --layers.w2v.gradWeight = addw2v
        --layers.w2v2.gradWeight = addw2v
      end

      return err, grads
    end

    -- gradient descent
    optim_method(func, params, config, state)
    -- reset padding embedding to zero
    layers.w2v.weight[1]:zero()
    layers.w2v2.weight[1]:zero()
    if opt.skip_kernel > 0 then
      -- keep skip kernel at zero
      layers.skip_conv.weight:select(3,3):zero()
    end

    -- Renorm (Euclidean projection to L2 ball)
    local renorm = function(row)
      local n = row:norm()
      row:mul(opt.L2s):div(1e-7 + n)
    end

    -- renormalize linear row weights
    local w = layers.linear.weight
    for j = 1, w:size(1) do
      renorm(w[j])
    end
    -- renormalize linear row weights
    local w_au = layers.linear_au.weight
    for j = 1, w_au:size(1) do
      renorm(w_au[j])
    end

    local w_au2 = layers.linear_au2.weight
    for j = 1, w_au2:size(1) do
      renorm(w_au2[j])
    end
  end
  
  for i = 1, test_shuffle:size(1) do
    local t = (test_shuffle[i] - 1) * opt.batch_size + 1
    local batch_size = math.min(opt.batch_size, test_size - t + 1)

    -- data samples and labels, in mini batches.
    local inputs = test:narrow(1, t, batch_size)
    local inputs_au = test_au:narrow(1, t, batch_size) --narrow(dim,index, size)
    local targets = test_label:narrow(1, t, batch_size)
    local targets_au = test_label_au:narrow(1, t, batch_size)    

    inputs = inputs:double()
    targets = targets:double()

    -- closure to return err, df/dx
    local func = function(x)
      -- get new parameters
      if x ~= params then
        params:copy(x)
      end
      -- reset gradients
      grads:zero()

      -- forward pass
      local output_all = model:forward({inputs, inputs_au})
      local outputs = output_all[1]
      local outputs_au = output_all[2] 
      local outputs_au2 = output_all[3]

      local err_au = criterion_au:forward(outputs_au, targets_au:select(2,1))
    --print('hehe')
      local err_au2 = criterion_au2:forward(outputs_au2, targets_au:select(2,2))

      -- track errors and confusion
      --total_err = total_err + err * batch_size
      for j = 1, batch_size do
        confusion4:add(outputs_au[j], targets_au[j]:select(2,1))
        confusion5:add(outputs_au2[j], targets_au[j]:select(2,2))
      end


      -- compute gradients
      local df_do = criterion:backward(outputs, targets)
      df_do:zero()
      local df_do_au = criterion_au:backward(outputs_au, targets_au:select(2,1))
      local df_do_au2 = criterion_au2:backward(outputs_au2, targets_au:select(2,2))
      model:backward({inputs,inputs_au}, {df_do,df_do_au,df_do_au2} )

      if opt.model_type == 'static' then
        -- don't update embeddings for static model
        layers.w2v.gradWeight:zero()
      elseif opt.model_type == 'multichannel' then
        -- keep one embedding channel static
        layers.chan1.gradWeight:zero()
      --elseif opt.model_type == 'nonstatic' then
        ---- don't update embeddings for static model
        --local addw2v = layers.w2v.gradWeight+layers.w2v2.gradWeight
        --layers.w2v.gradWeight = addw2v
        --layers.w2v2.gradWeight = addw2v
      end

      return err_au, grads
    end

    -- gradient descent
    optim_method(func, params, config, state)
    -- reset padding embedding to zero
    layers.w2v.weight[1]:zero()
    layers.w2v2.weight[1]:zero()
    if opt.skip_kernel > 0 then
      -- keep skip kernel at zero
      layers.skip_conv.weight:select(3,3):zero()
    end

    -- Renorm (Euclidean projection to L2 ball)
    local renorm = function(row)
      local n = row:norm()
      row:mul(opt.L2s):div(1e-7 + n)
    end

    -- renormalize linear row weights
    local w_au = layers.linear_au.weight
    for j = 1, w_au:size(1) do
      renorm(w_au[j])
    end

    local w_au2 = layers.linear_au2.weight
    for j = 1, w_au2:size(1) do
      renorm(w_au2[j])
    end
  end

  for i = 1, dev_shuffle:size(1) do
    local t = (dev_shuffle[i] - 1) * opt.batch_size + 1
    local batch_size = math.min(opt.batch_size, dev_size - t + 1)

    -- data samples and labels, in mini batches.
    local inputs = dev:narrow(1, t, batch_size)
    local inputs_au = dev_au:narrow(1, t, batch_size) --narrow(dim,index, size)
    local targets = dev_label:narrow(1, t, batch_size)
    local targets_au = dev_label_au:narrow(1, t, batch_size)    

    inputs = inputs:double()
    targets = targets:double()

    -- closure to return err, df/dx
    local func = function(x)
      -- get new parameters
      if x ~= params then
        params:copy(x)
      end
      -- reset gradients
      grads:zero()

      -- forward pass
      local output_all = model:forward({inputs, inputs_au})
      local outputs = output_all[1]
      local outputs_au = output_all[2] 
      local outputs_au2 = output_all[3]

      local err_au = criterion_au:forward(outputs_au, targets_au:select(2,1))
    --print('hehe')
      local err_au2 = criterion_au2:forward(outputs_au2, targets_au:select(2,2))

      -- track errors and confusion
      --total_err = total_err + err * batch_size
      for j = 1, batch_size do
        confusion4:add(outputs_au[j], targets_au[j]:select(2,1))
        confusion5:add(outputs_au2[j], targets_au[j]:select(2,2))
      end


      -- compute gradients
      local df_do = criterion:backward(outputs, targets)
      df_do:zero()
      local df_do_au = criterion_au:backward(outputs_au, targets_au:select(2,1))
      local df_do_au2 = criterion_au2:backward(outputs_au2, targets_au:select(2,2))
      model:backward({inputs,inputs_au}, {df_do,df_do_au,df_do_au2} )

      if opt.model_type == 'static' then
        -- don't update embeddings for static model
        layers.w2v.gradWeight:zero()
      elseif opt.model_type == 'multichannel' then
        -- keep one embedding channel static
        layers.chan1.gradWeight:zero()
      --elseif opt.model_type == 'nonstatic' then
        ---- don't update embeddings for static model
        --local addw2v = layers.w2v.gradWeight+layers.w2v2.gradWeight
        --layers.w2v.gradWeight = addw2v
        --layers.w2v2.gradWeight = addw2v
      end

      return err_au, grads
    end

    -- gradient descent
    optim_method(func, params, config, state)
    -- reset padding embedding to zero
    layers.w2v.weight[1]:zero()
    layers.w2v2.weight[1]:zero()
    if opt.skip_kernel > 0 then
      -- keep skip kernel at zero
      layers.skip_conv.weight:select(3,3):zero()
    end

    -- Renorm (Euclidean projection to L2 ball)
    local renorm = function(row)
      local n = row:norm()
      row:mul(opt.L2s):div(1e-7 + n)
    end

    -- renormalize linear row weights
    local w_au = layers.linear_au.weight
    for j = 1, w_au:size(1) do
      renorm(w_au[j])
    end

    local w_au2 = layers.linear_au2.weight
    for j = 1, w_au2:size(1) do
      renorm(w_au2[j])
    end
  end

  if opt.debug == 1 then
    print('Total err: ' .. total_err / train_size)
    print(confusion)
  end

  -- time taken
  time = timer:time().real - time
  time = opt.batch_size * time / train_size
  if opt.debug == 1 then
    print("==> time to learn 1 batch = " .. (time*1000) .. 'ms')
  end

  -- return error percent
  confusion:updateValids()
  confusion2:updateValids()
  confusion3:updateValids()
  confusion4:updateValids()
  confusion5:updateValids()
  return confusion.totalValid, confusion2.totalValid, confusion3.totalValid, confusion4.totalValid, confusion5.totalValid
end

function Trainer:test(test_data, test_data_au, test_labels, test_labels_au, model, criterion, criterion_au, criterion_au2)
  model:evaluate()

  local classes = {}
  for i = 1, opt.num_classes do
    table.insert(classes, i)
  end
  local num_classes = {}
  for i = 1, opt.num_classes_au do
    table.insert(num_classes, i)
  end
  local confusion = optim.ConfusionMatrix(classes)
  local confusion2 = optim.ConfusionMatrix(num_classes)
  local confusion3 = optim.ConfusionMatrix(num_classes)
  confusion:zero()
  confusion2:zero()
  confusion3:zero()

  local test_size = test_data:size(1)

  local total_err = 0

  for t = 1, test_size, opt.batch_size do
    -- data samples and labels, in mini batches.
    local batch_size = math.min(opt.batch_size, test_size - t + 1)
    local inputs = test_data:narrow(1, t, batch_size)
	  local inputs_au = test_data_au:narrow(1, t, batch_size)
    local targets = test_labels:narrow(1, t, batch_size)
    local targets_au = test_labels_au:narrow(1, t, batch_size)
    if opt.cudnn == 1 then
      inputs = inputs:cuda()
      targets = targets:cuda()
    else
      inputs = inputs:double()
      targets = targets:double()
    end

    local output_all = model:forward({inputs,inputs_au})
	  local outputs = output_all[1]
    local outputs_au = output_all[2] 
	  local outputs_au2 = output_all[3]
    local err = criterion:forward(outputs, targets)
    local err_au = criterion_au:forward(outputs_au, targets_au:select(2,1))
    local err_au2 = criterion_au2:forward(outputs_au2, targets_au:select(2,2))
    total_err = total_err + err * batch_size

    for i = 1, batch_size do
      confusion:add(outputs[i], targets[i])
      confusion2:add(outputs_au[i], targets_au[i]:select(2,1))
      confusion3:add(outputs_au2[i], targets_au[i]:select(2,2))
    end
  end

  if opt.debug == 1 then
    print(confusion)
    print('Total err: ' .. total_err / test_size)
  end

  -- return error percent
  confusion:updateValids()
  confusion2:updateValids()
  confusion3:updateValids()
  return confusion.totalValid, confusion2.totalValid, confusion3.totalValid
end

return Trainer
