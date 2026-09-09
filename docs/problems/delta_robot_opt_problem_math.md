# Delta Robot Trajectory Optimization Problem

## Geometry

Base radius $R_b$, platform radius $R_e$, bicep length $L_1$, forearm length $L_2$, combined offset $a = R_b - R_e$. Arm $j \in \{0,1,2\}$ sits at angle $\varphi_j = 90° + j\cdot120°$, with radial unit vector $u_j = (\cos\varphi_j, \sin\varphi_j, 0)$.

Arm $j$'s elbow, minus the platform-radius offset (the center of arm $j$'s forearm sphere):

$$
c_j(\theta_j) = \big(a + L_1\cos\theta_j\big)\,u_j + \big(0,0,-L_1\sin\theta_j\big)
$$

## Optimization variables

The optimization vector is obtained by stacking joint optimized variables in time, joint per joint, over $N$ fine-grid trajectory points:

$$
x = \big(\theta_{0,0},\dots,\theta_{0,N-1},\ \theta_{1,0},\dots,\theta_{1,N-1},\ \theta_{2,0},\dots,\theta_{2,N-1}\big) \in \mathbb{R}^{3N}
$$

$\theta_k = (\theta_{0,k},\theta_{1,k},\theta_{2,k})$ is the joint-angle triple at point $k$.

## Forward kinematics

$p(\theta) \in \mathbb{R}^3$ is the point equidistant ($L_2$) from all three $c_j(\theta_j)$, via trilateration:

$$
\lVert p - c_j(\theta_j) \rVert^2 = L_2^2, \qquad j=0,1,2
$$

Solved in closed form: subtract pairs of equations to get 2 linear equations in $p$ (a line of candidate solutions), intersect that line with one sphere (a scalar quadratic), and take the physical root (lower $z$).

**Jacobian**, by implicit differentiation of $F_j(p,\theta) = \lVert p-c_j(\theta_j)\rVert^2 - L_2^2 = 0$:

$$
\frac{\partial p}{\partial \theta} = -\left(\frac{\partial F}{\partial p}\right)^{-1}\frac{\partial F}{\partial \theta},
\qquad
\left[\frac{\partial F}{\partial p}\right]_{j,:} = 2(p-c_j)^\top,
\qquad
\frac{\partial F}{\partial \theta} = \mathrm{diag}\!\left(-2(p-c_j)\cdot\frac{dc_j}{d\theta_j}\right)
$$

a single $3\times3$ linear solve per point ($\partial F/\partial\theta$ is diagonal since $F_j$ depends only on $\theta_j$), verified against finite differences to $\sim\!10^{-7}$.

## Reference trajectory

The task is defined by a **tracked-target sequence** $\tau_0,\dots,\tau_M \in \mathbb{R}^3$ (an arbitrary ordered list of Cartesian points the end-effector must pass through, e.g. home plus the vertices of some polygon or star, in whatever order traces it) mapped onto tracked indices $K=\{0,9,18,\dots,9M\}\subset\{0,\dots,N-1\}$, i.e. $N=9M+1$: 8 untracked interior points between each consecutive pair of tracked targets, giving the optimizer room to shape a smooth transition. Nothing below depends on what $\tau_0,\dots,\tau_M$ actually trace: the formulation is the same whether it's a pentagon, a star, or any other path.

## Cost: nonlinear least squares (residuals nonlinear in $x$ through $p(\theta)$)

`furiaopt`'s `LSProblem` cost convention is $J(x) = F(x)^\top F(x)$ (no $\tfrac12$ factor; see `getApproximateHessian` in `src/direction_strategy.cpp`, which forms $2\nabla r\nabla r^\top$, and `cost_func_`/`gradient_func_` in `src/solvers/constrained_solver.cpp`, which form $r^\top r$ / $2\nabla r\, r$). The weights are folded into the residuals themselves ($\sqrt{w}\,(\cdot)$ below) so that squaring recovers exactly $w$:

$$
J(x) = w_{\text{track}}\sum_{k\in K} \big\lVert p(\theta_k) - \tau_k \big\rVert^2 \;+\; w_{\text{reg}}\sum_{k=0}^{N-2} \big\lVert p(\theta_{k+1}) - p(\theta_k) \big\rVert^2
$$

as residual blocks (each $\in\mathbb{R}^3$), stacked for `LSProblem`/Gauss-Newton:

$$
r_{\text{track},k} = \sqrt{w_{\text{track}}}\,\big(p(\theta_k)-\tau_k\big),\ k\in K
\qquad\qquad
r_{\text{reg},k} = \sqrt{w_{\text{reg}}}\,\big(p(\theta_{k+1})-p(\theta_k)\big),\ k=0,\dots,N-2
$$

### Full stacked residual vector $r(x) \in \mathbb{R}^{3(|K|+N-1)}$

$$
r(x) =
\begin{bmatrix}
\sqrt{w_{\text{track}}}\,(p(\theta_{k})-\tau_{k})_{k\in K} \\[4pt]
\sqrt{w_{\text{reg}}}\,(p(\theta_1)-p(\theta_0)) \\
\sqrt{w_{\text{reg}}}\,(p(\theta_2)-p(\theta_1)) \\
\vdots \\
\sqrt{w_{\text{reg}}}\,(p(\theta_{N-1})-p(\theta_{N-2}))
\end{bmatrix}
$$

The first $|K|$ blocks (each length 3) are the tracking residuals, one per tracked index; the remaining $N-1$ blocks are the smoothness residuals. The optimizer minimizes $J(x) = r(x)^\top r(x)$.

Residual Jacobian blocks (only $\theta_k$, or $\theta_k,\theta_{k+1}$, are touched, chained through $\partial p/\partial\theta$ above, no coupling across non-adjacent points):

$$
\frac{\partial r_{\text{track},k}}{\partial \theta_k} = \sqrt{w_{\text{track}}}\,\frac{\partial p}{\partial\theta}(\theta_k)
\qquad
\frac{\partial r_{\text{reg},k}}{\partial \theta_{k+1}} = \sqrt{w_{\text{reg}}}\,\frac{\partial p}{\partial\theta}(\theta_{k+1}),
\quad
\frac{\partial r_{\text{reg},k}}{\partial \theta_k} = -\sqrt{w_{\text{reg}}}\,\frac{\partial p}{\partial\theta}(\theta_k)
$$

Gauss-Newton Hessian approximation $\approx 2\,\nabla r\,\nabla r^\top$, only approximate (residuals are nonlinear in $x$), unlike a Cartesian-decision-variable design where it would be exact.

## Constraints

### Joint limits (box)

$$
\theta_{\min} \le \theta_{j,k} \le \theta_{\max}, \qquad \forall j\in\{0,1,2\},\ k=0,\dots,N-1
$$

As inequality constraints ($h(x)\ge 0$ convention): $h^{lo}(x) = x-\theta_{\min}\mathbf{1} \ge 0$, $h^{hi}(x) = \theta_{\max}\mathbf{1}-x \ge 0$; gradient is a constant $\pm$ Identity, no chain rule.

### Obstacle avoidance (circular keep-away, optional)

Any number of circular obstacles in the horizontal plane (vertical cylinders, so only $(x,y)$ matter). For obstacle $o$ with center $c_o\in\mathbb{R}^2$ and radius $r_o$, applied at every point $k=0,\dots,N-1$ (points far from an obstacle are simply slack):

$$
h_{o,k}(x) = \big\lVert p_{xy}(\theta_k) - c_o \big\rVert^2 - r_o^2 \ \ge\ 0
$$

a keep-away (one-sided) constraint: this is exactly why SQP is needed here rather than a plain QP, since the cost stays quadratic-like but this feasible region (complement of a disk) is nonconvex.

**Gradient**, via the chain rule through the same $\partial p/\partial\theta$ already derived above (only the $x,y$ rows of it are used, since the constraint doesn't depend on $z$):

$$
\frac{\partial h_{o,k}}{\partial \theta_k} = 2\big(p_{xy}(\theta_k) - c_o\big)^\top \left[\frac{\partial p}{\partial\theta}(\theta_k)\right]_{xy,:}
$$

a $1\times3$ row vector, nonzero only in $\theta_k$'s 3 columns (no coupling to other points, same locality as the joint-limit and residual gradients). Each obstacle adds $N$ rows to the inequality vector, on top of the $2\cdot3N$ joint-limit rows.
