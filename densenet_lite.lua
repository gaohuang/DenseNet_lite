require 'nn'
require 'cunn'
require 'cudnn'
require 'models/DenseLayer'

local function createModel(opt)
    if (opt.depth - 4 ) % 3 ~= 0 then
      error("Depth must be 3N + 4!")
    end

    --#layers in each denseblock
    local N = (opt.depth - 4)/3

    --growth rate
    local growthRate = 12

    --dropout rate, set it to nil to disable dropout, non-zero number to enable dropout and set drop rate
    local dropRate = nil

    --#channels before entering the first denseblock
    --set it to be comparable with growth rate
    local nChannels = 16

    local function addTransition(model, nChannels, growthRate, dropRate)
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))
      model:add(cudnn.SpatialConvolution(nChannels, growthRate, 1, 1, 1, 1, 0, 0))
      if dropRate then
        model:add(nn.Dropout(dropRate))
      end
      model:add(cudnn.SpatialAveragePooling(2, 2))
    end

    print("Building model")
    model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1))

    for i=1, N do
      model:add(nn.DenseLayer(i, nChannels, growthRate, 1))
      nChannels = nChannels + growthRate
    end
    model:add(nn.JoinTable(2))
    addTransition(model, nChannels, nChannels, dropRate)

    for i=1, N do
      model:add(nn.DenseLayer(i, nChannels, growthRate, 1))
      nChannels = nChannels + growthRate
    end
    model:add(nn.JoinTable(2))
    addTransition(model, nChannels, nChannels, dropRate)

    for i=1, N do
      model:add(nn.DenseLayer(i, nChannels, growthRate, 1))
      nChannels = nChannels + growthRate
    end
    model:add(nn.JoinTable(2))

    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialAveragePooling(8,8)):add(nn.Reshape(nChannels))
    if opt.dataset == 'cifar100' then
      model:add(nn.Linear(nChannels, 100))
    elseif opt.dataset == 'cifar10' then
      model:add(nn.Linear(nChannels, 10))
    else
      error("Dataset not supported yet!")
    end
    

    --Initialization following ResNet
    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
    end
    local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
    end

    local function DenseLayerInit(name)
      for k,denseLayer in pairs(model:findModules(name)) do
         v = denseLayer.net:get(1) -- BN
         v.weight:fill(1)
         v.bias:zero()     
         v = denseLayer.net:get(3) -- Conv
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
    end

    DenseLayerInit('nn.DenseLayer')
    ConvInit('cudnn.SpatialConvolution')
    ConvInit('nn.SpatialConvolution')
    BNInit('fbnn.SpatialBatchNormalization')
    BNInit('cudnn.SpatialBatchNormalization')
    BNInit('nn.SpatialBatchNormalization')
    for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
    end
    model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
