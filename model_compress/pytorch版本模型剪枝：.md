文件源码来自model-compression项目



pytorch版本模型剪枝：

* 拿到稀疏训练（L1正则后的VGG模型）

  ```Python
  #加载需要剪枝的模型，也即是稀疏训练得到的BaseLine模型，代码如下，
  # 其中args.depth用于指定VGG模型的深度，一般为16和19
  model = vgg(dataset=args.dataset, depth=args.depth)
  ```

  

* 确定剪枝的全局阈值，然后根据阈值得到剪枝之后的网络内层的通道数cfg_mask，这个cfg_mask可以确定剪枝后的模型的结构。这个过程只是确定每一层哪些索引的通道要被剪枝掉并获得cfg_mask，真正的剪枝操作还未执行

  ```python
  total = 0 #计算需要剪枝的变量个数total
  for m in model.modules():
      if isinstance(m, nn.BatchNorm2d):
          total += m.weight.data.shape[0]
  
  #确定剪枝的全局阈值
  bn = torch.zeros(total)
  index = 0
  for m in model.modules():
      if isinstance(m, nn.BatchNorm2d):
          size = m.weight.data.shape[0]
          bn[index:(index+size)] = m.weight.data.abs().clone()
          index += size
  
  # 按照权值大小排序
  sorted, indices = torch.sort(bn)
  thre_index = int(total * args.percent)
  # 确定要剪枝的阈值
  thre = sorted[thre_index]
  ```

  

* 预剪枝阶段

  ```python
  pruned = 0
  cfg = []
  cfg_mask = []
  for k, m in enumerate(model.modules()):
      if isinstance(m, nn.BatchNorm2d):
          weight_copy = m.weight.data.abs().clone()
          # 要保留的通道标记Mask图
          mask = weight_copy.gt(thre).float().cuda() #gt来比较a中元素大于b中元素，大于为1，不大于为0
          # 剪枝掉的通道数个数
          pruned = pruned + mask.shape[0] - torch.sum(mask)#所有权重数减去大于阈值权重的数量
          m.weight.data.mul_(mask)
          m.bias.data.mul_(mask)
          cfg.append(int(torch.sum(mask)))#保留通道数量
          cfg_mask.append(mask.clone())
          print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))
      elif isinstance(m, nn.MaxPool2d):
          cfg.append('M')
  
  pruned_ratio = pruned/total
  ```

  

* 预处理修剪后的简单测试模型（简单设置的BN缩放为零）

```Python
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

#调用来测试一下
acc = test(model)#model这里用的还是未剪枝的稀疏化模型
```

* 在开始剪枝前先回顾下Net的创建过程

  ```python
  class Net(nn.Module):
      def __init__(self, cfg = None):
          super(Net, self).__init__()
          if cfg is None:
              cfg = [192, 160, 96, 192, 192, 192, 192, 192]
          
          self.tnn_bin = nn.Sequential(
                  nn.Conv2d(3, cfg[0], kernel_size=5, stride=1, padding=2),
                  nn.BatchNorm2d(cfg[0]),
                  FP_Conv2d(cfg[0], cfg[1], kernel_size=1, stride=1, padding=0, first=1),
                  FP_Conv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
  
                  FP_Conv2d(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2),
                  FP_Conv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0),
                  FP_Conv2d(cfg[4], cfg[5], kernel_size=1, stride=1, padding=0),
                  nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
  
                  FP_Conv2d(cfg[5], cfg[6], kernel_size=3, stride=1, padding=1),
                  FP_Conv2d(cfg[6], cfg[7], kernel_size=1, stride=1, padding=0),
                  nn.Conv2d(cfg[7],  10, kernel_size=1, stride=1, padding=0),
                  nn.BatchNorm2d(10),
                  nn.ReLU(inplace=True),
                  nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                  )
          
          显然，Net由config文件控制，每层的卷积个数都在文件中明确写出，这给后面的卷积过程带来好处
  ```

  

* 开始剪枝

  ```python
  #根据之前预剪枝的cfg生成一个newmodel，它只指明了每层卷积的卷积核数量并未初始化
  newmodel = nin.Net(cfg)
  if not args.cpu:
      newmodel.cuda()
  layer_id_in_cfg = 0
  start_mask = torch.ones(3)
  end_mask = cfg_mask[layer_id_in_cfg]
  i = 0
  for [m0, m1] in zip(model.modules(), newmodel.modules()):#然后将新旧model合并
          if isinstance(m0, nn.BatchNorm2d):
          if i < layers - 1:
              i += 1
              idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
              if idx1.size == 1:
                  idx1 = np.resize(idx1, (1,))
              #在这一步，将卷积核中重要的权重全部copy给新的模型。
              m1.weight.data = m0.weight.data[idx1].clone()
              m1.bias.data = m0.bias.data[idx1].clone()
              m1.running_mean = m0.running_mean[idx1].clone()
              m1.running_var = m0.running_var[idx1].clone()
              layer_id_in_cfg += 1
              start_mask = end_mask.clone()
              if layer_id_in_cfg < len(cfg_mask):  
                  end_mask = cfg_mask[layer_id_in_cfg]
          else:
              m1.weight.data = m0.weight.data.clone()
              m1.bias.data = m0.bias.data.clone()
              m1.running_mean = m0.running_mean.clone()
              m1.running_var = m0.running_var.clone()
      elif isinstance(m0, nn.Conv2d):
          if i < layers - 1:
              idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
              idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
              if idx0.size == 1:
                  idx0 = np.resize(idx0, (1,))
              if idx1.size == 1:
                  idx1 = np.resize(idx1, (1,))
              w = m0.weight.data[:, idx0, :, :].clone()
              m1.weight.data = w[idx1, :, :, :].clone()
              m1.bias.data = m0.bias.data[idx1].clone()
          else:
              idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
              if idx0.size == 1:
                  idx0 = np.resize(idx0, (1,))
              m1.weight.data = m0.weight.data[:, idx0, :, :].clone()
              m1.bias.data = m0.bias.data.clone()
      elif isinstance(m0, nn.Linear):
              idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
              if idx0.size == 1:
                  idx0 = np.resize(idx0, (1,))
              m1.weight.data = m0.weight.data[:, idx0].clone()
  ```

  

* 剪枝后model测试

  ```python
  #******************************剪枝后model测试*********************************
  print('新模型: ', newmodel)
  print('**********剪枝后新模型测试*********')
  model = newmodel
  _test()
  #******************************剪枝后model保存*********************************
  print('**********剪枝后新模型保存*********')
  torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)
  print('**********保存成功*********\r\n')
  
  #*****************************剪枝前后model对比********************************
  print('************旧模型结构************')
  print(cfg_0)
  print('************新模型结构************')
  print(cfg, '\r\n')
  ```

  