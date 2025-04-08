# P2PNet人群计数模型

```cpp
git status
git commit -m "update: xxxx(2024-04-08)"
git push
```



# 模型特点

> ##### 基于**VGG16**作为骨干网络（Backbone）提取特征。

#### VGG16

一种**卷积神经网络**架构（最早由牛津大学的 Visual Geometry Group 提出，VGG)

> ##### 使用特征金字塔网络（FPN）融合多尺度特征。

#### 多尺度特征

在图像里，有的人可能很小（远处），有的人很大（近处），这时候需要：

- 低层（前几层卷积）：保留了更多细节，适合小物体

- 高层（后几层卷积）：保留了更多语义，适合大物体

### 特征图

**数据结构**

一个特征图就是一个`torch.Tensor` 对象，一般是[B,C,H,W]/[C,H,W]

**对应程序**

文件：`backbone.py`

```python
#输入图片 -> 特征图

xs = self.body(tensor_list)  # 输入图片 -> 特征图
return [xs]                  # 或多层特征图：[body1, body2, body3, body4]

'''
输出:
是一个或多个张量（Tensor），每个 shape 类似于：[batch_size, 256, H, W]
'''
```

文件：`p2pnet.py`

```python
'''
FPN 结构会：

把不同层的特征图（例如 C3, C4, C5）

融合成统一大小/语义的多尺度特征图（例如 P3, P4, P5）
'''
features = self.backbone(samples)# 返回一个 list：[C2, C3, C4, C5]
features_fpn = self.fpn([features[1], features[2], features[3]])#把C3, C4, C5融合

#输出：这些融合后的特征图仍然是 torch.Tensor，只是包含更多信息。
```

`features` 是从 backbone（VGG16 + 提取层）中提取出来的中间结果

它是一个 **list of torch.Tensor**

```python
features = [C1, C2, C3, C4]
```

其中每个 Cn 都是一个 **特征图**

它们是从 **VGG16 的每一层卷积模块（conv block）** 输出的特征图。

| 编号 | 层级    | 输出尺寸（假设输入为224×224） | 特征语义           |
| ---- | ------- | ----------------------------- | ------------------ |
| C1   | conv1_2 | `[64, 112, 112]`              | 较低级（边缘）     |
| C2   | conv2_2 | `[128, 56, 56]`               | 低级（纹理）       |
| C3   | conv3_3 | `[256, 28, 28]`               | 中级（局部结构）   |
| C4   | conv4_3 | `[512, 14, 14]`               | 高级（物体部件）   |
| C5   | conv5_3 | `[512, 7, 7]`                 | 更高级（语义目标） |

> ##### 使用分类与回归两个分支预测人群数量与位置。

#### 分类分支

预测anchor point是不是人(给出一个二分类概率（是不是“人”）)

#### 回归分支

预测人群的精确位置坐标(给出预测的人（或目标）坐标相对于某个 Anchor Point 的偏移量**，换句话说回归给出的是偏移量**)

`p2pnet.py`211行

```python
output_coord = regression + anchor_points
```

`regression`：回归分支输出的偏移量（Δx, Δy），单位为像素

`anchor_points`：每个 anchor point 的坐标，是固定的、预先生成好的

`output_coord`：模型最终预测的人群坐标（实际位置）

#### anchor point

`p2pnet.py`

```python
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points
```

`stride=16`：每个特征图像素对应原图 16×16

`row = 3,line = 3`：均匀分布`row  * line`个anchor point

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

------

### `p2pnet.py`

- **功能**：
  - 定义 P2PNet 模型结构：
    - 回归分支预测点坐标
    - 分类分支预测是否为前景
    - 使用 anchor points + FPN + backbone
  - 定义 `SetCriterion_Crowd`：用于计算点位置与分类的损失
- **输入**：
  - 图像张量（已转换为 `NestedTensor`）
- **输出**：
  - 模型输出字典：
    - `pred_logits`: 预测的前景概率，`[B, N, 2]`
    - `pred_points`: 预测坐标，`[B, N, 2]`

------

### `backbone.py`

- **功能**：
  - 定义骨干网络（目前支持 VGG16/VGG16-BN）
  - 用于提取图像特征
- **输入**：
  - 输入图像张量 `[B, 3, H, W]`
- **输出**：
  - 多层特征图（可用于 FPN）

------

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

# 程序用到的一些库/函数

### torch.Tensor

| 创建方式                         | 张量内容         | `.shape` 结果           | 维度说明                 |
| -------------------------------- | ---------------- | ----------------------- | ------------------------ |
| `torch.tensor(3.14)`             | 单个数           | `torch.Size([])`        | 0维张量（标量）          |
| `torch.tensor([1, 2, 3])`        | 一维数组         | `torch.Size([3])`       | 1维张量（向量）          |
| `torch.tensor([[1, 2], [3, 4]])` | 矩阵（二维数组） | `torch.Size([2, 2])`    | 2维张量（矩阵）          |
| `torch.randn(3, 4, 5)`           | 3维张量          | `torch.Size([3, 4, 5])` | 像一个 3 层楼的 4×5 表格 |

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

### .shape/.unsqueeze(0)

三维时

[channels, height, width]

四维时

[batch_size=4, channels=3, height=224, width=224]

`4`张（`224` * `224`的`3`通道彩色图片）

`channels`：灰度图的通道数是 1，RGB 彩色图的通道数是 3

```python
t = torch.randn(4, 3, 224, 224)
print(t.shape)  #torch.Size([4, 3, 224, 224])
```

