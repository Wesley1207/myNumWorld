# 问题: 1D wave equation

$$
\begin{aligned}
\frac{\partial^{2} u}{\partial t^{2}} &=c^{2} \frac{\partial^{2} u}{\partial x^{2}} + f(x,t), & x \in(0, L), & t \in(0, T] \\
u(x, 0) &=I(x), & x & \in[0, L] \\
\frac{\partial}{\partial t} u(x, 0) &= V(x), & x & \in[0, L] \\
u(0, t) &=0, & t & \in(0, T] \\
u(L, t) &=0, & t & \in(0, T]
\end{aligned}
$$


# 离散化

$$
\frac{u_i^{n+1} - 2u_i^n + u_i^{n-1}}{\Delta t ^2} = c^2 \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{\Delta x^2} + f_i^n
$$

$$
   u_i^{n+1} = c^2 \frac{\Delta t^2}{\Delta x^2} u_{i-1}^n - u_i^{n-1} + (2 - 2c^2 \frac{\Delta t^2}{\Delta x^2}) u_i^n + c^2 \frac{\Delta t^2}{\Delta x^2} u_{i+1}^n + \Delta t^2 f_i^n
$$

即, 
$$
u_{i}^{n+1}=-u_{i}^{n-1}+2 u_{i}^{n}+C^{2}\left(u_{i+1}^{n}-2 u_{i}^{n}+u_{i-1}^{n}\right) + \Delta t^2 f_i^n
$$
其中 $C= c\frac{\Delta t}{\Delta x}$ 是 **Courant number**. 在波动方程中, Courant number 是关键参数, 并且 Courant number 是无量纲的。

当 $n=0$ 时, 我们会遇到 $u_i^{-1}$。求解该项的方法是利用初始条件中的导数项。即,

$$
D_{2t}u_i^n = \frac{u_{i}^{n+1} - u_{i}^{n-1}}{2\Delta t} = V(x_i)
$$
当$n=0$时, 有:
$$
D_{2t}u_i^0 = \frac{u_{i}^{1} - u_{i}^{-1}}{2\Delta t} = V(x_i)
$$
即:
$$
u_i^{-1} = u_i^1 - 2\Delta t V_i
$$
把上式代入离散后的波动方程表达式, 有:
$$
u_{i}^{1}=-u_i^1 + 2\Delta t V_i+2 u_{i}^{0}+C^{2}\left(u_{i+1}^{0}-2 u_{i}^{0}+u_{i-1}^{0}\right) + \Delta t^2 f_i^0 \\
u_{i}^{1}= \Delta t V_i+ u_{i}^{0}+\frac{C^{2}}{2}\left(u_{i+1}^{0}-2 u_{i}^{0}+u_{i-1}^{0}\right) + \frac{1}{2}\Delta t^2 f_i^0 \\
$$


# 构造解析解测试收敛率

## 构造解析解

我们解析解的约束是: $u(0,t)=u(L,t)=0$, 于是我们选择解析解的形式为:
$$
u_e(x,t)  = x(L-x) \mathrm{sin}t
$$

把 $u_e$ 代入原方程$u_{tt}=c^2 u_{xx}+f$, 得到
$$
f(x,t) = 2c^2 \mathrm{sin} t-(L-x)x\mathrm{sin}t
$$

得到初始条件为:
$$
u(x,0) = 0 \\
u_t(x, 0) = x(L-x)
$$

当选用以上方程的时候, 我们可以方便地测试算法的正确性。

## 收敛率定义

为了测试收敛率 (convergence rate), 我们需要构造一系列的算例, 让每个算例的时间或者空间网格都比上一个算例的网格更精细。收敛率的测试依赖于假设:

$$
E = C_t \Delta t^r + C_x \Delta x^p
$$
其中 $E$ 是数值误差, $C_t$, $C_x$, $r$ 和 $p$ 是常数。$r$ 是时间上的收敛率, $p$ 是空间上的收敛率。对于我们采用的中心差分的离散方法, 我们期待 $r=p=2$ (根据截断误差的推导)。

通常来说, $C_t$ 和 $C_x$ 的值的大小, 我们是不关心的。

还有一种误差的简化表示法。根据 Courant number $C= c\frac{\Delta t}{\Delta x}$, 并令 $h=\Delta t$, 我们有:

$$
E = C_t \Delta t^r + C_x \Delta x^r = C_t h^r + C_x \left(\frac{c}{C}\right)^r h^r = Dh^r \\
D = C_t + C_x \left(\frac{c}{C}\right)^r
$$

## 计算误差

首先, 定义一个初始的 $h_0$, 然后定义随后的$h$:

$$
h_i = 2^{-i} h_0, \quad i=0,1,2,...,m
$$
对于每一个算例, 我们都记录 $E$ 和 $h$。 常见的 $E$ 有两种选择方法, $\ell^2$ norm 或者 $\ell^\infty$ norm:

$$
\begin{aligned}
&E=\left\|e_{i}^{n}\right\|_{\ell^{2}}=\left(\Delta t \Delta x \sum_{n=0}^{N_{t}} \sum_{i=0}^{N_{x}}\left(e_{i}^{n}\right)^{2}\right)^{\frac{1}{2}}, \quad e_{i}^{n}=u_{\mathrm{e}}\left(x_{i}, t_{n}\right)-u_{i}^{n} \\
&E=\left\|e_{i}^{n}\right\|_{\ell^{\infty}}=\max _{i, n}\left|e_{n}^{i}\right|
\end{aligned}
$$

另外一种方式是记录单一时间步骤上的误差 $\ell^2$ 或 $\ell^\infty$, 比如在最后一个时间步上 ($n=N_t$):

$$
\begin{aligned}
&E=\left\|e_{i}^{n}\right\|_{\ell^{2}}=\left(\Delta x \sum_{i=0}^{N_{x}}\left(e_{i}^{n}\right)^{2}\right)^{\frac{1}{2}}, \quad e_{i}^{n}=u_{\mathrm{e}}\left(x_{i}, t_{n}\right)-u_{i}^{n}, \\
&E=\left\|e_{i}^{n}\right\|_{\ell^{\infty}}=\max _{0 \leq i \leq N_{x}}\left|e_{i}^{n}\right| .
\end{aligned}
$$


## 计算收敛率

令 $E_i$ 和 $h_i$ 对应于相应算例的误差和时间步, 则: $E_i = Dh_i^r$, 针对两次连续的算例, 有: 
$$
E_{i+1} = Dh_{i+1}^r \\
E_i = Dh_i^r
$$
两次算例的 $E_i$ 相除以消去 $D$, 有:
$$
r = \frac{\ln\left( \frac{E_{i+1}}{i}\right)}{\ln \left(\frac{h_{i+1}}{h_i}\right)}
$$

对于 $0,1,...,m$ 这样 $m+1$ 个算例, 一共有 $m$ 个$r_i, \quad i=0,1,...,m-1$。对于当前的波动方程, 中心差分, 我们期待随着 $i$ 的增加 $r_i$ 收敛到 $2$.


# Implementation

实践算法的时候, 应该扫描 Courant number $C$, 即保持 $\Delta t$ 不变, 根据 $C$ 改变 $\Delta x$。




