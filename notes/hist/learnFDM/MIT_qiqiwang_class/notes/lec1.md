# Convection-diffusion equation

## form

$$
\frac{\partial u}{\partial t} + U \frac{\partial u}{\partial x} = \kappa \frac{\partial ^2 u}{\partial x^2} + f
$$

一阶导数项是对流项, 二阶导数项是扩散项。$f$ 描述外部流量的影响。

## 特例: 热传导方程

热传导方程对应于对流-扩散方程的 $U\rightarrow0, f\rightarrow 0$:

$$
\frac{\partial u}{\partial t} = \kappa \frac{\partial ^2 u}{\partial x^2}
$$

### 傅里叶分析

$$
u(k) = \sum_{k=-\infty}^{k=\infty} \hat{u}_k e^{ikx}
$$

这样, 傅里叶分解之后, 周期解对导数的性质就特别好: 
$$
\frac{\partial u(x)}{\partial x} = ik u(x)
$$


## 特例: 椭圆方程

elliptic eqn 对应于对流-扩散方程的 $U\rightarrow 0, t\rightarrow \infty$:

$$
\frac{\partial ^2 u}{\partial x^2} + f = 0
$$

## 特例: 双曲方程

hyperbolic eqn: 对应于对流-扩散方程的 $\kappa \rightarrow 0, f \rightarrow 0$:

$$
\frac{\partial u}{\partial t} + U \frac{\partial u}{\partial x} = 0
$$