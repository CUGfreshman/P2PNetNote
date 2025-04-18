# P2PNet人群计数模型

**conda python3.7**

```
conda create -n p2p37 python=3.7 -y
```

**PyTorch 1.7.1 + CUDA 11.0**

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

**其他依赖**

```
pip install tensorboardX easydict pandas numpy==1.19.2 scipy==1.5.2 matplotlib Pillow opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```





```cpp
git status
git add .
git commit -m "update: xxxx(2024-04-08)"
git push
```

【【精读AI论文】VGG深度学习图像分类算法】https://www.bilibili.com/video/BV1fU4y1E7bY?p=9&vd_source=3471d188fd1d74b6d33be0296ab20feb

# P2PNet模型特点

## 模型处理流程

```txt
samples (image)
   ↓
VGG Backbone 提取多层特征 → features = [C3, C4, C5]
   ↓
Decoder (FPN)
   ↓
features_fpn = P3_x (最终融合特征)
   ↓
feed into regression + classification branches
   ↓
输出点的位置（回归）+ 点的概率（分类）
回归分支输出：[B,H*W*anchor,2]即每个锚点的坐标偏移
分类分支输出：[B,H*W*anchor,1]即每个锚点的概率
```

```py
#backbone: 骨干网络（通常是VGG16）
#row,line： 就是anchor那的参数

class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
       #有两类
        self.num_classes = 2                  
       #每个patch的锚点数量
        num_anchor_points = row * line         
       
       #创建回归和分类分支
        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256, \
                                            num_classes=self.num_classes, \
                                            num_anchor_points=num_anchor_points)

       #锚点
        self.anchor_points = AnchorPoints(pyramid_levels=[3,], row=row, line=line)
       
       #FPN
        self.fpn = Decoder(256, 512, 512)
```

```py
#Samples：输入数据，类型为NestedTensor，包含图像张量（samples.tensors）和掩码（samples.mask）

    def forward(self, samples: NestedTensor):

        features = self.backbone(samples)#[C1_2, C3, C4, C5]
        
        features_fpn = self.fpn([features[1], features[2], features[3]])#[P3_x, P4_x, P5_x]

        batch_size = features[0].shape[0]

        #运行回归和分类分支
        regression = self.regression(features_fpn[1]) * 100 #[B, H*W*4, 2]。这里偏移量乘以 100
        classification = self.classification(features_fpn[1])#[B, H*W*4, 1]
        
        #生成锚点坐标，并重复 batch_size 次以匹配批量大小
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)#[B,H*W*anchor,2]每个锚点的坐标
        
        #将回归分支预测的偏移量 regression 加到锚点坐标 anchor_points 上，得到最终的点坐标。
        output_coord = regression + anchor_points
        
        #将分类分支的输出 classification 直接作为最终的分类结果
        output_class = classification
        
        #组织输出,作为模型的输出
        out = {'pred_logits': output_class, 'pred_points': output_coord}
       
        return out
```



## VGG16

VGG16输入：[3,224,224]图片

![VGG16](.\image\VGG16.png)

**注意block1/2/3/4/5包括了：Conv + maxpool**

**注意block1/2/3/4/5还可以叫做C1/2/3/4/5**

![QQ20250410-183439](.\image\QQ20250410-183439.png)

**Conv3-64**指：64个3*3\*【通道数】的卷积核（卷积层的深度自动与输入的通道数匹配）。3\*3\*3的卷积核可以理解为对每一'层'[1,254,254]运用3\*3卷积核，得到3个[1,254,254]再相加。

**FC-4096**：`4096`个神经元的`全连接层(FC)`

经过maxpool变成[64,112,112]。**池化层降低分辨率**

每一层卷积之后通常都会接一个非线性激活函数（比如ReLU）

## FPN

本项目中，FPN的唯一输入就是输入经VGG16网络的C3,C4,C5的中间结果（是特征图）

**VGG的不同层C3,C4,C5**就是FPN所谓的多尺度特征。

低层特征：小尺度，找小人头（远处的人头）

高层特征：大尺度，找大人头（近处的人头）

**下采样：**指分辨率降低。比如8倍下采样就是[B,C,H,W]->[B,C,H/8,W/8]。通道数变化不知道。但CNN一般会在下采样的同时扩大通道数。

`__init__`

```python
#Cx_size 指通道数
#feature_size=256 : FPN 会把所有特征图的通道数统一成 256

#FPN的全部内容
class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        #第一步：准备处理 C5 → P5
        #1. 把 C5 从 512通道降到 256
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        #2. 把 C5 的分辨率放大一倍,让它的大小和 C4（下一层地图）一样，方便叠加
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #3. 进一步处理放大后的特征
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        #第二步：准备处理 C4 → P4
        #1. 把 C4 转成相同通道数（256）
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        #2. 再放大一倍，准备和更低层的 C3 地图融合
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #3. 进一步处理放大后的特征
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        #第三步：准备处理 C3 → P3
        #C3本来就是256通道？注意不同的VGG16也许有区别，不用太在意这里。
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
```

`forward`

```py
def forward(self, inputs):
        C3, C4, C5 = inputs
        
        #先对C5降通道（通道统一是256），上采样（跟C4一样），卷积再处理一下
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        #然后对P4降通道，P4 = C4 + 上采样后的 P5，上采样，卷积再处理一下
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        #降通道，P3 = C3 + 上采样后的 P4，卷积再处理一下
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]
    #注：返回几个融合程度不同的特征图，通道数都是256，但分辨率依次下降
    
#至于这里为什么用P5/4_upsampled_x进行融合而不是P5/4_x：
#多尺度融合时，应尽可能用原始的深层语义特征，直接参与融合，不要“先加工再融合”
```

## 分类分支

预测anchor point是不是人(给出一个二分类概率（是不是“人”）)

```py
#num_features_in：输入特征图的通道数
#num_anchor_points=4：每个小块（patch）内的锚点数量
#num_classes=80：分类的类别数，默认为 80
#Prior=0.01 ：一个先验概率（prior）
#feature_size=256：特征中间图的通道数

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()
```

```py
#x是[B,C,H,W]

def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)#[B,8,H,W]。8 = 4(anchor) * 2(class)

        out1 = out.permute(0, 2, 3, 1)#[B,H,W,8]

        #下面进行reshape
        
        #获取维度信息
        batch_size, width, height, _ = out1.shape
        
        #维度重塑 [B,H,W,4,2]
        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)#[B,H*W*anchor,2]
        #输出的最后一个维度的2，表示[有人头的概率，无人头的概率]
```



## 回归分支

```py
#4个卷积层（带 ReLU 激活函数）和 1 个输出卷积层

#num_features_in：输入特征图的通道数(按照程序的输入流，应该是从FPN的输出来的)
#num_anchor_points=4：每个小块（patch）里的锚点数量
#feature_size=256：中间特征图的通道数

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)
        #num_anchor_points * 2的乘2是：每个锚点需要预测 2 个值（x 和 y 方向的偏移量）
        #看作每个 [h, w] 的 patch 里有 num_anchor_points 个点，每个点要预测两个值：(Δx, Δy)
```

```py
 # x：输入特征图，通常来自 FPN（特征金字塔网络），形状是[B,C,H,W]
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)#回归预测在这里

        
        #reshape
        out = out.permute(0, 2, 3, 1)# [B, H, W, 8]
        
        return out.contiguous().view(out.shape[0], -1, 2)# [B, H*W*4, 2]
        #H*W*4是所有anchor点的数目，2的x和y
```

## anchor point

```py
#特征金字塔的层级：表示在哪些层生成锚点
#每个层级的步长（stride）:不指定就是2^层级

class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line
```

```py
#image：[B,3,H,W]
  def forward(self, image):
        image_shape = image.shape[2:]#[H,W]
        image_shape = np.array(image_shape)#转换为 NumPy 数组。
        
        #计算每个层级（pyramid_levels）的特征图大小
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        #image_shapes是[[H,W],[H,W],[H,W]....]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        #创建一个空的 NumPy 数组，用来存储所有锚点的坐标（0行2列）
        
        
        for idx, p in enumerate(self.pyramid_levels):#遍历每个层级
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)#生成锚点位置模版(一个patch里)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)#平移到对应位置
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)#把shifted_anchor_points添加到 all_anchor_points

         #添加批维度[B,N,2]
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
       
        #锚点坐标从 NumPy 格式变成 PyTorch 格式
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))
```

```python
#这个函数其实是在输出锚点分布的格式模版

#在一个stride×stride 的小块（patch）中,生成row×line 网格的锚点
#默认参数为 stride=16, row=3, line=3，即在一个 16×16像素的小块中生成 3×3=9个锚点
def generate_anchor_points(stride=16, row=3, line=3):
    #1. 计算锚点的间距
    row_step = stride / row      #每行锚点之间的间距
    line_step = stride / line    #每列锚点之间的间距
    
    #2. 计算锚点的偏移量（相对于小块中心的坐标）
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    #np.arange(1, line + 1)生成一个NumPy数组，从1到line + 1
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    
    #3. 将 x 和 y 的偏移量组合成所有可能的坐标对。x和y现在都是row * line的二维矩阵
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    #4. 改变形状
    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()
    #shift_x.ravel() 和 shift_y.ravel() 将网格坐标展平为一维数组。
    #np.vstack 堆叠 x 和 y 坐标，生成形状为row * line的pair数组

    return anchor_points
    #返回一个形状为row * line的pair数组，表示小块内所有锚点的坐标。
```

```py
#shift是作用域某一层特征图。把锚点模板（一组预设的偏移量）搬到特征图的每个像素位置，生成所有锚点的实际坐标。
#shape是特征图的大小[H,W]
#stride指明一个patch有多大
#锚点位置模版

def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points
```



> ##### 使用匈牙利算法（Hungarian Matcher）匹配预测点与真实点。

> ##### 整个模型训练过程使用 Adam 优化器和交叉熵损失以及位置回归损失







# 程序模块

## 各个模块详解

### `engine.py`

- **功能**：
  - 提供训练 (`train_one_epoch`) 和评估 (`evaluate_crowd_no_overlap`) 函数。
  - 包含损失计算、梯度更新与模型权重更新。
- **输入**：
  - `model`：模型实例（如 P2PNet）
  - `criterion`：损失函数
  - `data_loader`：训练或验证集 DataLoader
  - `optimizer`：优化器
- **输出**：
  - 训练时返回损失指标字典。
  - 评估时返回 `(mae, mse)`（平均绝对误差、均方误差平方根）。

------

### `run_test.py`

- **功能**：
  - 加载训练好的模型进行推理（测试单张图片）。
- **输入**：
  - `--weight_path`：模型权重路径
  - `--output_dir`：保存预测图像的目录
  - 图片路径（代码中写死为 `./vis/demo1.jpg`）
- **输出**：
  - 带预测点的可视化图片保存在 `output_dir` 中。

```py
from PIL import Image

img_path = "./vis/demo1.jpg"

img_raw = Image.open(img_path).convert('RGB')
#img_raw 用来存储加载后的图像对象
#convert('RGB')将图像转换成3通道

width, height = img_raw.size
#img_raw.size返回含两个整数（宽度和高度）的元组

new_width = width // 128 * 128
new_height = height // 128 * 128
#先整数除128再乘以128：图形大小变成128的整数倍

img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
#Image.ANTIALIAS是一个滤波器
```



------

### `train.py`

- **功能**：
  - 主训练入口，包含模型构建、数据集加载、训练过程与周期性评估。
- **输入**：
  - 多个命令行参数（如学习率、batch_size、数据集路径、模型结构等）
- **输出**：
  - 日志文件（`run_log.txt`）
  - TensorBoard 可视化日志（默认保存到 `./runs`）
  - 检查点文件：`latest.pth`、`best_mae.pth`

```python
from crowd_datasets import build_dataset
loading_data = build_dataset(args=args)#args是命令行参数
train_set, val_set = loading_data(args.data_root)
```

------

### `loading_data.py`

- **功能**：
  - 构建训练集和验证集（使用 `SHHA` 类）。
- **输入**：
  - `data_root`：数据根目录路径
- **输出**：
  - `(train_set, val_set)`：PyTorch Dataset 对象

------

### `SHHA.py`

- **功能**：
  - 定义 `SHHA` 数据集类，支持训练与测试模式。
  - 实现数据加载、增强（裁剪、缩放、翻转）。
- **输入**：
  - `data_root`：数据目录
  - `train`：是否为训练集
  - `transform`：图像预处理方法
- **输出**：
  - 图像张量 `img`：Tensor, `[C, H, W]`
  - 目标列表 `target`：每个元素为一个字典，含 `point`（点坐标）、`image_id` 和 `labels`

```py
class SHHA(Dataset)#继承自PyTorch 的 Dataset类
```



**SetCriterion_Crowd**

**功能**：损失计算器，包含分类（交叉熵）+ 回归（MSE）

`loss_labels()`：计算分类损失

`loss_points()`：计算坐标回归损失

`get_loss()`：调度上述两个

`forward()`：综合计算损失（对一个 batch）

```py
class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_bbox.sum() / num_points

        return losses

def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        if len(indices1) == 0:
           print("⚠️ 没有有效的匹配结果，跳过这个 batch")
           return {"loss_ce": torch.tensor(0.0, device=output1["pred_logits"].device)}

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses

# create the P2PNet model
def build(args, training):
    # treats persons as a single class
    num_classes = 1

    backbone = build_backbone(args)
    model = P2PNet(backbone, args.row, args.line)
    if not training: 
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(num_classes, \
                                matcher=matcher, weight_dict=weight_dict, \
                                eos_coef=args.eos_coef, losses=losses)

    return model, criterion
```



------

### `backbone.py`

- **功能**：
  - 定义骨干网络（目前支持 VGG16/VGG16-BN）
  - 用于提取图像特征
- **输入**：
  - 输入图像张量 `[B, 3, H, W]`
- **输出**：
  - 多层特征图（可用于 FPN）

```py
#backbone:一个预构建好的 VGG 网络   num_channels:希望提取出来的特征图的通道数
#name:指定使用的 VGG 模型名称，比如 'vgg16_bn' 或 'vgg16'   
#return_interm_layers:布尔值，用来控制是否返回中间层的特征图。如果是 True，就会把 VGG 网络分割成若干部分，分别提取；如果 False，则只保留最后的一个整体输出。
def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        #backbone.features 是 VGG 网络中专门提取卷积层部分的成员（通常是一个 nn.Sequential 对象）
        #而 children() 方法返回其中所有子模块（每一层）的迭代器
        #features 就是一个列表，里面存放了 VGG 网络中所有卷积、池化、激活等层
        
        
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])   #C1,2
                self.body2 = nn.Sequential(*features[13:23]) #C3
                self.body3 = nn.Sequential(*features[23:33]) #C4
                self.body4 = nn.Sequential(*features[33:43]) #C5
            else:
                self.body1 = nn.Sequential(*features[:9])    #C1,2
                self.body2 = nn.Sequential(*features[9:16])  #C3
                self.body3 = nn.Sequential(*features[16:23]) #C4
                self.body4 = nn.Sequential(*features[23:30]) #C5
        else:                                                #打包整个c1-5
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 输出会16倍下采样
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 输出会16倍下采样
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
```

别忘了，这里的C1/2/3/4/5就是block1/2/3/4/5

**可以把C/body理解为一个函数，输入为[B,C,H,W]/[C,H,W]输出为[B,C,H,W]/[C,H,W]（特征图）**

### `vgg_.py`

- **功能**：
  - 定义 VGG 系列网络结构（11/13/16/19 层）
  - 支持加载 ImageNet 预训练模型
- **输入**：
  - 图像张量 `[B, 3, H, W]`
- **输出**：
  - 特征图张量 `[B, C, H', W']`

------

### `matcher.py`

- **功能**：
  - 使用匈牙利算法实现预测点与真实点的一一匹配
- **输入**：
  - `outputs`: 模型输出（包含 logits 和点坐标）
  - `targets`: Ground Truth（包含 labels 和点坐标）
- **输出**：
  - 每个 batch 匹配结果的索引元组 `(index_i, index_j)`

------

### `misc.py`

- **功能**：
  - 实用工具集合，包含：
    - 日志记录器 `MetricLogger`
    - 分布式训练辅助函数
    - 数据整理函数 `collate_fn_crowd`
    - 损失函数 `FocalLoss`
- **输入/输出**：
  - 因函数众多，具体视调用情况而定

------

### `__init__.py`（模型部分）

- **功能**：
  - 定义统一接口 `build_model(args, training=False)`
- **输入**：
  - `args`：包含参数配置的对象
- **输出**：
  - 若 `training=True`：返回 `(model, criterion)`
  - 否则仅返回 `model`

------

### `__init__.py`（数据部分）

- **功能**：
  - 根据参数 `dataset_file` 加载对应数据集（目前支持 `SHHA`）
- **输入**：
  - `args.dataset_file`
- **输出**：
  - 加载函数 `loading_data`（不是数据本身）

## 调用关系示意图

```txt
train.py / run_test.py
      │
      ├─ engine.py（训练/推理）
      │       └── matcher.py (匹配算法)
      │
      ├─ loading_data.py (加载数据集)
      │       └── SHHA.py (数据增强与加载具体逻辑)
      │
      └─ build_model (模型构建)
            └── p2pnet.py (P2PNet整体网络结构)
                    ├─ backbone.py（特征提取）
                    │      └─ vgg_.py (VGG网络)
                    └─ misc.py (辅助函数)

```

## 整体运行流程

训练时 (`train.py`)：

1. 构建数据集 (loading_data.py + SHHA.py)。
2. 构建模型 (`build_model` -> `p2pnet.py`)。
3. 开始训练 (`engine.py` 中的 `train_one_epoch`)。
4. 每隔固定 epoch 评估模型性能 (`evaluate_crowd_no_overlap`)。
5. 保存最佳模型权重与日志记录。

推理时 (`run_test.py`)：

1. 加载模型与预训练权重。
2. 单张图片进行预测与可视化。

# 程序数据流

## 读入->(img,target)

一张jpg文件+一个标记了人头位置的txt文件，首先读入程序，以下面的方式存储

```py
img.shape → [3, H, W]
target = {
    'point': [[x1, y1], [x2, y2], ..., [xN, yN]],  # 所有标注的人头坐标（像素位置）
    'labels': [1, 1, ..., 1],  # 所有人头对应的类别标签（全是 1）
    'image_id': tensor([i])  # 第 i 张图,图像编号
}

#SHHA类的def __getitem__就是返回 return img, target
```

## （img，target）被打包成（samples，targets）

第一步构造的哪些(img,target)会打包成以下结构。

**打包：img->samples**

**打包：target->targets**

```py
samples: NestedTensor（包含图像张量 [B, 3, H, W],以及一个[B,H,W]表明图片哪些地方是padding，为了把所有图片都扩充到H*W所添加的假像素）
targets: List[dict]，长度为 B，每个 dict 是 target
```

## samples被送入模型（VGG16/FPN)->输出pred_logits, pred_points

samples输入到P2PNet的forward

```py
def forward(self, samples: NestedTensor):
        #第一步：提取图像张量
        features = self.backbone(samples)
        #features是list：[C3, C4, C5] 
        
        
        #第二步：送入 FPN
        features_fpn = self.fpn([features[1], features[2], features[3]])
        #features_fpn是list：[P3, P4, P5]
        
        
        batch_size = features[0].shape[0]
        #第三步：回归 以及 分类
        regression = self.regression(features_fpn[1]) * 100 # 8x
        #regression是坐标偏移[B, N, 2]：Tensor
        #每个 anchor point 预测一个 (Δx, Δy)
        classification = self.classification(features_fpn[1])
        #classfication是输出每个 anchor 的概率[B, N, 2]：Tensor
        #其中：N 就是每张图中 anchor point 的数量
        
        #第四步：生成 anchor 点位置
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        #anchor_points是[B, N, 2]，即每个 anchor 的参考坐标
        
        
        #第五步：偏移 + anchor = 预测坐标
        output_coord = regression + anchor_points
        output_class = classification
        
        
        out = {'pred_logits': output_class, 'pred_points': output_coord}
       '''
       out = {
             'pred_logits': Tensor [B, N, 2],  # 每个 anchor 点的分类概率
             'pred_points': Tensor [B, N, 2],  # 每个 anchor 点的预测位置坐标 (x, y)
}
       '''
        return out
```



# 程序用到的一些库/函数

### torch.Tensor

| 创建方式                         | 张量内容 | `.shape` 结果           | 维度说明                       |
| -------------------------------- | -------- | ----------------------- | ------------------------------ |
| `torch.tensor(3.14)`             | 单个数   | `torch.Size([])`        | 0维张量（标量）                |
| `torch.tensor([1, 2, 3])`        | 一维数组 | `torch.Size([3])`       | 1维张量（向量）                |
| `torch.tensor([[1, 2], [3, 4]])` | 2维数组  | `torch.Size([2, 2])`    | 2维张量（矩阵）                |
| `torch.randn(3, 4, 5)`           | 3维数组  | `torch.Size([3, 4, 5])` | 三维数组（理解为三层，每层4x5) |

**程序**

**`SHHA.py`**

```python
point[i] = torch.Tensor(point[i])
img = torch.Tensor(img)
```

**`run_test.py`**

```python
samples = torch.Tensor(img).unsqueeze(0)
#.unsqueeze(0)添加batch维度，比如[3, 128, 128]->[1, 3, 128, 128]
```



### ToTensor()

转换成PyTorch Tensor

**程序中**

**`loading_data.py`**, **`run_test.py`**

```python
transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(...)#归一化到 [0, 1] 区间
])
```

### nn.Conv2d

```python
nn.Conv2d(
    in_channels,     # 输入通道数
    out_channels,    # 输出通道数（卷积核个数）
    kernel_size=3,   # 卷积核大小：3×3
    padding=1        #特征图周围加 1 像素的填充，确保卷积后特征图的空间大小不变。
)
```

### DataSet

```py
torch                # 顶级模块：PyTorch 的主包
└── utils            # torch.utils：工具模块（utils）
    └── data         # torch.utils.data：数据处理相关
        └── Dataset  # Dataset 类：数据集抽象基类
```

```py
因为在 PyTorch 中，所有能送进 DataLoader 的数据集，必须是 Dataset 的子类，并实现两个函数：

__len__()：告诉你总共有多少个样本

__getitem__(index)：告诉你怎么通过索引取出第 index 个样本
```









# 代码阅读顺序

------

### 🥇 **第一阶段：模型结构理解（推荐优先）**

> 🌟先弄清楚“模型是如何处理一张图，输出预测的”。

1. **`p2pnet.py`**
    🔍 理解 P2PNet 的模型结构、前向传播逻辑、回归/分类输出
2. **`backbone.py` + `vgg_.py`**
    🔍 看 backbone 是如何提取特征图的（用的是 VGG16）
3. **`matcher.py`**
    🔍 看预测点和真实点是如何一一匹配的（匈牙利算法）
4. **`misc.py`**（看 `NestedTensor`, `collate_fn_crowd`）
    🔍 输入图片是如何被打包、送进模型的

------

### 🥈 **第二阶段：训练与推理流程理解**

> 🔧 理解“模型是怎么训练的、如何评估/推理”。

1. **`engine.py`**
    🔍 训练和评估时模型的调用逻辑，损失如何计算与反传
2. **`train.py`**
    🔍 主训练入口：怎么加载模型 + 数据，如何控制每轮训练与保存
3. **`run_test.py`**
    🔍 推理流程：如何加载模型、处理图片、可视化预测结果

------

### 🥉 第三阶段：数据流 & 数据集部分

> 📦 理解“模型的输入是什么？Ground Truth 从哪里来？”

1. **`loading_data.py`**
    🔍 创建数据集 + transform 的入口函数
2. **`SHHA.py`**
    🔍 读图片 + 标注点，做数据增强（裁剪、缩放、翻转）
3. **`__init__.py`（模型 & 数据）**
    🔍 提供统一的构建接口，做跳转引用用的
