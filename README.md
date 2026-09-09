# FuriaOpt

**This project is now in a good state to be used, but it is subject to continuous improvements**

FuriaOpt is a C++ optimization library designed to solve NLP, LS, QP, and LP problems, with or without linear and non-linear constraints.

<div align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center;">
  <img src="docs/images/sqp_delta_robot_problem_4_star_two_obstacles_trajectory.gif" width="45%">
  <img src="docs/images/qp_trajectory-cropped.png" width="45%">
</div>

<div align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center;">
  <img src="docs/images/rosenbrock-cropped.png" width="45%">
</div>

## Features
*   **Solvers:** NLP (Gradient Descent, BFGS, Exact Newton), LS (Gauss-Newton), QP/LP (primal log-barrier interior-point method), and SQP for nonlinearly-constrained NLP/LS.
*   **Constraints:** Supports both linear and non-linear constraints.
*   **Testing:** Includes a comprehensive suite of unit tests.
*   **Examples:** Demonstrations on how to integrate and use the solvers.
*   **Visualizer:** To visualize iterations of the solvers on provided examples.

## Requirements
*   C++20 or higher
*   CMake
*   Eigen 3
*   Catch2 (for testing)
*   spdlog (for logging)
*   nlohmann_json (for config loading)

## AI Usage Disclosure
During development, Claude Code (Anthropic Opus and Sonnet family models) was used as an engineering assistant for:
*   Code REVIEW of the solver components, no direct modifications.
*   Generation of unit and integration tests.
*   Refinement of examples and python visualizers.

The fundamental concepts, algorithms, and implementation strategy originate from the author's prior knowledge, refer to the background section.

## Background
- Constrained Numerical Optimization for Estimation and Control (CNOEC), Lorenzo Fagiano, Politecnico di Milano.

## Examples

### 1. Nonlinear Constrained Least-Squares Problem - SQP
Delta robot joint-space trajectory optimization uses a parallel, closed-chain kinematic structure: the three actuated joints jointly determine the end-effector position through a nonlinear, closed-form forward-kinematics map, so the tracking cost itself is nonlinear in the decision variables. Obstacle avoidance is added directly as a constraint on that same optimization. The nonlinear cost functions and constraints make it a suitable problem to solve using the SQP method.

For mathematical derivation of the optimization problem refer to [delta robot opt problem](docs/problems/delta_robot_opt_problem_math.md).

<div align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center;">
  <img src="docs/images/sqp_delta_robot_problem_2_pentagon_two_obstacles_trajectory.gif" width="45%">
  <img src="docs/images/sqp_delta_robot_problem_2_pentagon_two_obstacles_convergence.png" width="45%">
</div>

### 2. Unconstrained Non-Linear Problem
The chosen example for Unconstrained NLP problem solving is the classic Rosenbrock problem, comparing Gradient Descent, BFGS, Exact Newton and Gauss-Newton all from the same random starting point:

$$f(x_0, x_1) = (a - x_0)^2 + b(x_1 - x_0^2)^2, \quad a = 1,\ b = 100$$

<div align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center;">
  <img src="docs/images/rosenbrock.png">
</div>

Gauss-Newton performs especially well here because Rosenbrock is naturally a sum of squares whose residuals are at most quadratic in x, so there's no higher-order nonlinearity for the linearized-residual approximation to miss.

### 3. Quadratic Program with Linear Constraints
Smooths a 2D path pinned at unevenly-spaced waypoints, confined to a straight corridor around the start-end line:

$$\min_{u}\ \tfrac12\sum_{d\in\{x,y\}}\lVert Du_d\rVert_2^2 \quad\text{s.t.}\quad u_k = p_k\ (k\in\mathcal A), \quad -w_r \le n^\top u_k \le w_\ell\ \ \forall k$$

$D$ is the discrete 2nd-difference operator, $(Du)_i = u_{i-1} - 2u_i + u_{i+1}$, a finite-difference approximation of acceleration at unit time steps; $A=\{0,10,30,49\}$ are the pinned waypoints, $n$ the corridor normal. We don't supply $x_0$: the solver computes its own strictly feasible starting point via an internal phase-1 QP.

<div align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center;">
  <img src="docs/images/qp_trajectory.png">
</div>

## Execute the package

*Tested only on Ubuntu 24.04*

See [Getting Started](docs/getting_started.md) for build, run, and visualization instructions.

## License
This project is licensed under the [MIT License](LICENSE).