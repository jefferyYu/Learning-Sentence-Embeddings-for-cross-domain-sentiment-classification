require 'torch'
require 'nn'
require 'nngraph'

local ModelBuilder = torch.class('ModelBuilder')

function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    print('haha')
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    print('hehe')
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function ModelBuilder:make_net(w2v)
  if opt.cudnn == 1 then
    require 'cudnn'
    require 'cunn'
  end

  local input = nn.Identity()()
  local input_au = nn.Identity()()

  local lookup
  if opt.model_type == 'multichannel' then
    local channels = {}
    for i = 1, 2 do
      local chan = nn.LookupTable(opt.vocab_size, opt.vec_size)
      chan.weight:copy(w2v)
      chan.weight[1]:zero() --padding should always be 0
      chan.name = 'channel' .. i
      table.insert(channels, chan(input))
    end
    lookup = channels
  else
    lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
	  lookup_au = nn.LookupTable(opt.vocab_size, opt.vec_size)
    share_params(lookup, lookup_au)
    if opt.model_type == 'static' or opt.model_type == 'nonstatic' then
      lookup.weight:copy(w2v)
      --lookup_au.weight:copy(w2v)
    else
      -- rand
      lookup.weight:uniform(-0.25, 0.25)
      lookup_au.weight:uniform(-0.25, 0.25)
    end
    -- padding should always be 0
    lookup.weight[1]:zero()
    --lookup_au.weight[1]:zero()
    input_lookup = lookup(input)
	  input_lookup_au = lookup_au(input_au)
  end

  -- kernels is an array of kernel sizes
  local kernels = opt.kernels
  local layer1 = {}
  local layer_au1 = {}
  for i = 1, #kernels do
    local conv
    local conv_layer
    local max_time
    if opt.cudnn == 1 then
      conv = cudnn.SpatialConvolution(1, opt.num_feat_maps, opt.vec_size, kernels[i])
      if opt.model_type == 'multichannel' then
        local lookup_conv = {}
        for chan = 1, 2 do
          table.insert(lookup_conv, nn.Reshape(opt.num_feat_maps, opt.max_sent-kernels[i]+1, true)(
            conv(
            nn.Reshape(1, opt.max_sent, opt.vec_size, true)(
            lookup[chan]))))
        end
        conv_layer = nn.CAddTable()(lookup_conv)
        max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
      else
        if opt.highway_conv_layers > 0 then
          -- Highway conv layers
          local highway_conv = HighwayConv.conv(opt.vec_size, opt.max_sent, kernels[i], opt.highway_conv_layers)
          conv_layer = nn.Reshape(opt.num_feat_maps, opt.max_sent-kernels[i]+1, true)(
            conv(nn.Reshape(1, opt.max_sent, opt.vec_size, true)(
            highway_conv(lookup))))
          max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
        else
          conv_layer = nn.Reshape(opt.num_feat_maps, opt.max_sent-kernels[i]+1, true)(
            conv(
            nn.Reshape(1, opt.max_sent, opt.vec_size, true)(
            lookup)))
          max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
        end
      end
    else
      conv = nn.TemporalConvolution(opt.vec_size, opt.num_feat_maps, kernels[i])
      if opt.model_type == 'multichannel' then
        local lookup_conv = {}
        for chan = 1,2 do
          table.insert(lookup_conv, conv(lookup[chan]))
        end
        conv_layer = nn.CAddTable()(lookup_conv)
        max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
      else
        conv = nn.TemporalConvolution(opt.vec_size, opt.num_feat_maps, kernels[i])
        conv_au = nn.TemporalConvolution(opt.vec_size, opt.num_feat_maps2, kernels[i])
        --share_params(conv, conv_au)
        conv_layer = conv(input_lookup)
        conv_layer_au = conv_au(input_lookup_au)
        max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
        max_time_au = nn.Max(2)(nn.ReLU()(conv_layer_au)) -- max over time
      end
    end

    conv.weight:uniform(-0.01, 0.01)
    conv.bias:zero()
    conv_au.weight:uniform(-0.01, 0.01)
    conv_au.bias:zero()
    table.insert(layer1, max_time)
	  table.insert(layer_au1, max_time_au)
	
  end

  if opt.skip_kernel > 0 then
    -- skip kernel
    local kern_size = 5 -- fix for now
    local skip_conv = cudnn.SpatialConvolution(1, opt.num_feat_maps, opt.vec_size, kern_size)
    skip_conv.name = 'skip_conv'
    skip_conv.weight:uniform(-0.01, 0.01)
    -- skip center for now
    skip_conv.weight:select(3,3):zero()
    skip_conv.bias:zero()
    local skip_conv_layer = nn.Reshape(opt.num_feat_maps, opt.max_sent-kern_size+1, true)(skip_conv(nn.Reshape(1, opt.max_sent, opt.vec_size, true)(lookup)))
    table.insert(layer1, nn.Max(3)(cudnn.ReLU()(skip_conv_layer)))
  end

  local conv_layer_concat
  if #layer1 > 1 then
    conv_layer_concat = nn.JoinTable(2)(layer1)
	  conv_layer_concat_au = nn.JoinTable(2)(layer_au1)
  else
    conv_layer_concat = layer1[1]
	  conv_layer_concat_au = layer_au1[1]
  end

  local last_layer = conv_layer_concat
  local last_layer_au = conv_layer_concat_au
  if opt.highway_mlp > 0 then
    -- use highway layers
    local highway = HighwayMLP.mlp((#layer1) * opt.num_feat_maps, opt.highway_layers)
    last_layer = highway(conv_layer_concat)
  end

  -- simple MLP layer
  local linear = nn.Linear((#layer1) * opt.num_feat_maps + (#layer_au1) * opt.num_feat_maps2, opt.num_classes)
  linear.weight:normal():mul(0.01)
  linear.bias:zero()
  
  local linear_au = nn.Linear((#layer_au1) * opt.num_feat_maps2, opt.num_classes_au)
  linear_au.weight:normal():mul(0.01)
  linear_au.bias:zero()
  
  local linear_au2 = nn.Linear((#layer_au1) * opt.num_feat_maps2, opt.num_classes_au)
  linear_au2.weight:normal():mul(0.01)
  linear_au2.bias:zero()

  local softmax
  local softmax_au
  local softmax_au2
  softmax = nn.LogSoftMax()
  softmax_au = nn.LogSoftMax()
  softmax_au2 = nn.LogSoftMax()


  local output = softmax(linear(nn.Dropout(opt.dropout_p)(nn.JoinTable(2)({last_layer, last_layer_au})))) 
  local output_au = softmax_au(linear_au(nn.Dropout(opt.dropout_p)(last_layer_au))) 
  local output_au2 = softmax_au2(linear_au2(nn.Dropout(opt.dropout_p)(last_layer_au))) 
  model = nn.gModule({input, input_au}, {output, output_au, output_au2})
  return model
end

return ModelBuilder
