# safety RL

## Environment

1. Point-Circle (Source : Constrained Policy Optimization)

## Implementations:

### PPO-Lagrangian

 A CMDP is denoted by the tuple $(S, A, R, C, \gamma, \mu)$

$S_0,A_0,R_1,C_{1},S_1,A_1,R_2,C_{2},S_2,A_2,R_3,C_{3}\cdots$

Objective function:

$$
J^C\left(\pi_\theta\right)=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t C\left(s_t, a_t, s_{t+1}\right) \mid s_0 \sim \mu, a_t \sim \pi_\theta, \forall t\right]
$$

The (cost) constraint function:

$$
J^R\left(\pi_\theta\right)=\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R\left(s_t, a_t, s_{t+1}\right) \mid s_0 \sim \mu, a_t \sim \pi_\theta, \forall t\right] .
$$

The constrained optimization problem:
$$
\max _\theta J^R\left(\pi_\theta\right) \text { s.t. } J^C\left(\pi_\theta\right) \leq d
$$

A parameter θ will be called a feasible point if the cost constraint is satisfied for θ, i.e., $J^C (π_θ) ≤ d$.
$$
L(\theta, \lambda)=J^R\left(\pi_\theta\right)-\lambda\left(J^C\left(\pi_\theta\right)-d\right)
$$

Given a function:

$$
\theta_p(\theta)=\min_{\lambda:\lambda\geq0}L(\theta,\lambda)
$$

$\bullet$ If $J^{C}(\pi_{\theta})>d$:

$$
\theta_p(\theta)=\min_{\lambda:\lambda\geq0}[J^R(\pi_\theta)-\lambda J^C(\pi_\theta)] = -\infty
$$

$\bullet$ If $J^{C}(\pi_{\theta})\leq d$:

$$
\theta_p(\theta)=\min_{\lambda:\lambda\geq0}[J^R(\pi_\theta)-\lambda J^C(\pi_\theta)] = J^R(\pi_\theta)
$$

Which is equivalent to:

$$
\max_\theta\theta_p(\theta)=\max_\theta\min_{\lambda:\lambda\geq0}L(\theta,\lambda)=\max_\theta J^R(\pi_\theta)
$$

Equivalent objective function:

$$
L\left(\theta^*, \lambda^*\right)=\max _\theta \min _\lambda L(\theta, \lambda)
$$

Solving the max-min problem as above is equivalent to finding a global optimal saddle point $(θ^∗, λ^∗)$

Using Gradient search:

$$
\begin{aligned}
\theta_{n+1} &=\theta_n-\eta_1(n) \nabla_{\theta_n}\left(-L\left(\theta_n, \lambda_n\right)\right) \\
&=\theta_n+\eta_1(n)\left[\nabla_{\theta_n} J^R\left(\pi_\theta\right)-\lambda_n \nabla_{\theta_n} J^C\left(\pi_\theta\right)\right]
\end{aligned}
$$

$$
\begin{aligned}
\lambda_{n+1} &=\left[\lambda_n+\eta_2(n) \nabla_{\lambda_n}\left(-L\left(\theta_n, \lambda_n\right)\right)\right]_{+} \\
&=\left[\lambda_n-\eta_2(n)\left(J^C\left(\pi_\theta\right)-d\right)\right]_{+}
\end{aligned}
$$

Here$ [x]_{+}$ denotes $\max(0, x)$, $η_1(n), η_2(n) > 0\  ∀n$ are certain prescribed step-size schedules.

Estimation of $J^R\left(\pi_\theta\right)$, $J^C\left(\pi_\theta\right)$

$$
J^R\left(\pi_\theta\right)=\mathbb{E}_t\left[\min \left(r_t(\theta) A_t^R, \operatorname{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) A_t^R\right)\right]
$$

$$
J^C\left(\pi_\theta\right)=\mathbb{E}_t\left[\min \left(r_t(\theta) A_t^C, \operatorname{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) A_t^C\right)\right]
$$

Value functions loss and cost value functions loss:

$$
\operatorname{Loss}\left(\psi_r\right)=\frac{1}{M|T|} \sum_{\tau \in M} \sum_{t=0}^T\left(V_{\psi_r}^R\left(s_t\right)-\hat{R}_t\right)^2
$$

$$
\operatorname{Loss}\left(\psi_c\right)=\frac{1}{M|T|} \sum_{\tau \in M} \sum_{t=0}^T\left(V_{\psi_c}^C\left(s_t\right)-\hat{C}_t\right)^2
$$

 where $A_t^R=\sum_{l=0}^k(\gamma \bar{\lambda})^l \delta_{t+l}^R$,   and $A_t^C=\sum_{l=0}^k(\gamma \bar{\lambda})^l \delta_{t+l}^C$.  M is  mini-batch of size ≤ NT.

Let $δ^R_t = r_{t+1} + V^ R_ {ψ_r} (s_{t+1}) − V^R_{ψ_r} (s_t)$ and $δ^C_t = c_{t+1} + V^C_{ψ_c} (s_{t+1}) − V^ C_{ ψ_c} (s_t)$,

This approach is known as PPO-Lagrangian

## Results

![截屏2022-10-29 16.15.43](README.assets/%E6%88%AA%E5%B1%8F2022-10-29%2016.15.43.png)

## Reference

- PPO Lagrangian Reproduction in Pytorch https://github.com/akjayant/PPO_Lagrangian_PyTorch

- Safe Reinforcement Learning Using Advantage-Based Intervention https://github.com/nolanwagener/safe_rl
- Safety Gym https://github.com/openai/safety-gym
