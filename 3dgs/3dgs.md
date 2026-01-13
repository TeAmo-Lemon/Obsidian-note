# 高斯椭球
## 1. 核心定义式
$$G(x) = \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$
其中：

- $x \in \mathbb{R}^3$ 是空间中的三维坐标点。
    
- $\mu \in \mathbb{R}^3$ 是高斯体的中心位置（Mean）。
    
- $\Sigma$ 是 $3 \times 3$ 的正定协方差矩阵（Covariance Matrix），决定了高斯体的形状和旋转。
## 2. 协方差矩阵的分解 (可微分参数化)
由于 $\Sigma$ 必须保持[[正定性]]，直接优化它非常困难。在 3DGS 中，将其分解为**旋转矩阵 $R$** 和**缩放矩阵 $S$**：

$$\Sigma = RSS^T R^T$$

其中：

- **$S$ (Scaling)**：对角阵 $\text{diag}(s_x, s_y, s_z)$，表示在三个主轴方向上的缩放。
    
- **$R$ (Rotation)**：由单位四元数 $q$ 转换而来的旋转矩阵。
## 3. 投影到 2D 图像平面
投影后的 2D 协方差矩阵 $\Sigma'$ 定义为：

$$\Sigma' = J W \Sigma W^T J^T$$

其中：

- $W$ 是世界坐标系到相机坐标系的变换矩阵（Viewing transformation）。
    
- $J$ 是仿射变换的雅可比矩阵。
---
**线性变换下的协方差传递**

首先，我们需要一个概率论中的基础结论：

若随机变量 $X \sim N(\mu, \Sigma)$，对其进行一个线性变换 $Y = Ax + b$，那么变换后的随机变量 $Y$ 仍然服从高斯分布，其新的协方差矩阵为：

$$\Sigma_{new} = A \Sigma A^T$$

---

**分解相机成像过程**

将 3D 高斯投影到屏幕，经历两次坐标变换：

#### 第一步：世界坐标系 $\to$ 相机坐标系 (Affine Transformation)

相机的位置和姿态由变换矩阵 $W$（通常包含旋转 $R$ 和平移 $t$）定义。这是一个线性变换。

根据上面的传递性质，高斯体在相机空间下的协方差为：

$$\Sigma_{cam} = W \Sigma W^T$$
高斯分布的仿射变换（这里b不起作用）
$$ \mathbf{w} = A\mathbf{x} + b $$
$$\mathbf{w} \sim \mathcal{N}\bigl(A\mu + b,\; A\Sigma A^T\bigr)$$
#### 第二步：相机坐标系 $\to$ 图像坐标系 (Projective Transformation)

透视投影（Perspective Projection）本身是非线性的。

点 $(x, y, z)$ 投影到像平面的公式通常是 $u = f \cdot \frac{x}{z}$。因为分母里有 $z$，所以它不是一个简单的矩阵乘法。

---

**利用雅可比矩阵（Jacobian）进行局部线性化**

为了能继续使用 $A \Sigma A^T$ 这个简洁的公式，3DGS 采用了 **“局部线性化”** 的思想：

在投影中心点 $\mu_{cam}$ 附近，我们可以用一个线性矩阵 $J$ 来近似这个非线性的投影过程。

$$J = \frac{\partial \text{Project}(x)}{\partial x} \Big|_{x=\mu_{cam}}$$

$J$ 描述了：当你在相机空间移动极小的距离时，投影到屏幕上的点移动了多少。

因此，将投影过程近似为线性变换后，再次应用协方差传递公式：

$$\Sigma' = J (\Sigma_{cam}) J^T$$

代入第一步得到的 $\Sigma_{cam}$，就得到了最终公式：

$$\Sigma' = J W \Sigma W^T J^T$$
## 最终渲染公式(Alpha Blending)
对于图像上的某个像素，其颜色 $C$ 是通过对重叠在该像素上的 $N$ 个高斯体按深度排序后进行混合得到的：
$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

其中：

- $c_i$ 是第 $i$ 个高斯体的颜色。
    
- $\alpha_i$ 是该高斯体在像素处的有效不透明度，由预设的不透明度 $\sigma_i$ 和 2D 高斯值计算得出：
    
    $$\alpha_i = \sigma_i \cdot \exp\left(-\frac{1}{2}(x'-\mu')^T (\Sigma')^{-1} (x'-\mu')\right)$$


