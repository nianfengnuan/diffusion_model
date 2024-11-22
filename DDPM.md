# Denoising Diffusion Probabilistic Models

## 生成模型

### 定义

* 一个能随机生成与训练数据一致的模型

**问题**

* 如何对训练数据建模
* 如何采样

**思路**

* 从一个简单分布采样是容易的
* 从简单分布到观测数据分布是可以拟合的

**解题思路**

* 将观测数据分布映射到简单分布**【Encoder】**
* 将简单分布中映射到观测数据分布**【Decoder】**

**一个复杂的分布可以用多个高斯分布来表示**

假设有**K**个高斯分布，这K个高斯分布称作混合模型的隐变量则复杂分布的概率分布是：
$$
P_{\theta}=\sum_{i=1}^{K}P(z_{i})*P_{\theta}(x\vert z_{i})
$$
这里$P(z_{i})$表示第i个高斯分布在观测数据中所占概率

$P_{\theta}(x\vert z_{i})$表示第i个高斯分布的概率分布函数



一个复杂的分布可以用多个高斯分布来表示

![image-20241118110157276](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20241118110157276.png)

假设有K个高斯分布，这K个高斯分布称为混合模型的隐变量，则复杂分布的概率分布是：
$$
P_{\theta}=\sum_{i=1}^{K}P(z_{i})*P_{\theta}(x\vert z_{i})
$$
式中$p_{z_{i}}$表示第i个高斯分布在观测数据中所占概率，$P_{\theta}(x\vert z_{i})$表示第i个高斯分布的概率分布函数

将上式离散表示转换为连续(将z变成连续变量)表示：
$$
P_{\theta}=\int P(z)\cdot P_{\theta}(x\vert z)
$$
### Diffusion vs VAE

![image-20241120172721526](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20241120172721526.png)

**Diffusion model包含两个过程**

**前向扩散过程**

向观测数据中逐步加入噪声，直到观测数据变成高斯分布

**反向生成过程**

从一个高斯分布中采样，逐步消除噪声，直到变成清晰数据

![image-20241120172445431](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20241120172445431.png)

###

## **前向扩散过程**

### 如何加噪声

$$
x_{t}=\sqrt{\beta_{t}}*x_{t-1}+\sqrt{1-\beta_{t}}*\varepsilon_{t-1}
$$

### 加了多少次噪声

在DDPM中一共加入了1000次噪声



### 重参数采样(为什么weight带根号)

若y是一个高斯分布$y\sim N(\mu,\sigma^2)$,则$\frac{y-\mu}{\sigma}\sim N(0,1)$

设$\varepsilon$为一个标准高斯分布，则$y=\sigma*\varepsilon+\mu \sim N(\mu,\sigma^2)$

因此$x_{t}$满足高斯分布，且$x_{t}\sim N(\beta_{t}*x_{t-1},\sqrt{1-\beta_{t}})$

**在DDPM中，$\beta_{t}$是随着t线性减小的（随着t的增大，$x_{t}$趋近于标准高斯分布）**

### 实现流程图

![image-20241120174153990](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20241120174153990.png)

## 反向生成过程

**从一个高斯分布采样，通过反转过程生成观测图像**

![image-20241118211234438](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20241118211234438.png)

$p_{\theta}$是要生成的观测图像，通过不断对初始图像$p(x_{T})$乘高斯分布生成观测图像。

### 实现流程图

![image-20241120175617136](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20241120175617136.png)

![image-20241121211516697](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20241121211516697.png)

### 极大似然估计（Maximum Likelihood Estimation MLE）

********

**用于估计统计模型参数的方法，其核心思想是找到使观测数据出现概率最大的参数值。**

#### 基本原理

假设观测数据$x={x_{1},x_{2},...,x_{n}}$是来自某个概率分布$f(x\vert \theta)$的独立分布样本，其中$\theta$是需要估计的参数。

1. **似然函数：**似然函数表示给定参数$\theta$下，观测到数据x的联合概率密度:
   $$
   L(\theta \vert x)=\prod_{i=1}^{n}f(x_{i}\vert \theta)
   $$

2. **对数似然函数：**为了方便计算，我们通常对似然函数取对数:

$$
\ln L(\theta\vert x)=\sum_{i=1}^{n}\ln f(x_{i}\vert\theta)
$$

3. **极大化**：通过优化(求导数并验证)使上式达到最大值，找到使数据出现最可能的参数$\hat{\theta}$:
   $$
   \hat{\theta}=\arg \max_{\theta}\ln L(\theta\vert x)
   $$
   

### KL散度（Kullback-Leibler Divergence）

************

**这是一个用来衡量两个概率分布之间差异的统计量。**

假设$P$和$Q$是两个概率分布：

* $P(x)$：真实分布（或目标分布）
* $Q(x)$:   近似分布（或参考分布）

KL散度的数学定义为：
$$
D_{KL}(P\parallel Q)=\sum P(X)\log\frac{P(x)}{Q(x)}
$$
或者：
$$
D_{KL}(P\parallel Q)=\int P(x)\log\frac{P(x)}{Q(x)}dx
$$

所有影像生成模型的最终目标就是Maximum Likelihood =Minimize KL Divergence

### FID（Fréchet Inception Distance)

******

**用于评估生成模型性能的指标，特别是在图像生成任务中**

#### 定义

基于Inception网络计算两个图像集的特征表示，通过比较这些特征的统计量来评估生成图像的质量。

#### 计算方法

- 首先，通过Inception网络（如Inception V3）计算生成图像集和真实图像集的特征向量。

- 然后，计算每个集合的特征向量的均值（$\mu$）和协方差矩阵（$\sum$）。

- 最后，使用以下公式计算FID分数$\text{FID} = \|\mu_g - \mu_r\|^2_2 + \text{Tr}(\Sigma_g + \Sigma_r - 2(\Sigma_g\Sigma_r)^{1/2})$

  $\mu_{g}$和$\mu_{r}$分别是生成图像和真实图像的均值向量，$\sum g$ 和 $\sum r$分别是生成图像和真实图像的协方差矩阵，$Tr$ 表示矩阵的迹（即矩阵对角线元素的和）。

#### 物理含义

* FID测量了两个多元高斯分布之间的Fréchet距离，反映了真实图像分布和生成图像分布在高维特征空间中的相似程度。
* 较低的FID分数表示生成的图像在视觉质量上接近真实图像，并且在多样性上也与真实图像分布相似。
