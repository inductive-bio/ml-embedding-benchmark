# Setup + Background

The Bradley-Terry model, first proposed (CITE) has recently become a popular model for ranking preferences. The basic model assumes each item being compared has some underlying latent ability, call it $w_i$ for item $i$. When items $i, j$ face off, the probability of item $i$ winning is given by: 

$$
P(i \succ j) = \frac{w_i}{w_i + w_j}
$$

We cannot directly observe the latent abilities, but we can observe samples from this distribution. A helpful reparameterization involves defining $\beta_i = \log w_i$ so we can write:

$$
P(i\succ j) = \sigma(\beta_i - \beta_j)
$$

The Bayesian Bradley-Terry model (CITE) was developed to rank the abilities of machine learning models across datasets. 

Define $W_{ij}$ as the number of ‘wins’ model $i$ gets against model $j$. Then the likelihood function is: 

$$
L(w) = \prod_{i, j=1}^M (\frac{w_i}{w_i+w_j})^{W_{ij}}
$$

and we compute $w^* = \{w_1, w_2, ..., w_n\} = \argmax_w L(w)$.

The Bayesian model is based on: 

$$
W_{ij} \sim \text{Binomial}(N_{ij}, \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}) \\ \beta_i \sim \mathcal{N}(0, \sigma^2) \\ \sigma \sim \text{LogNormal}(0, 0.5)
$$

However, note this likelihood function does not model the true joint distribution, as all pairs are not independent. Once we know $i \succ j$ and $j \succ k$, $i \succ k$ naturally follows. The variance inflation this causes can be computed exactly to be a factor of $\frac{M+1}{3}$ where $M$ is the number of models being compared (see appendix).

# Introducing Margins

## Theoretical Framework

Testing models on various datasets yields more information than simply “model $i$ beats model $j$,” we also know the margin by which model $i$ beats model $j$. For our case, the dataset will be various endpoints, so indexing into endpoints is equivalent to indexing into datasets for the general case. We will define the margin by which $i$ beats $j$ on endpoint $e$:

$$
d_{ije} \equiv y_{je} - y_{ie}
$$

$y_{ie}$ is the population log MAE of model $i$ on endpoint $e$. Every endpoint on this benchmarking study is already transformed with log base 10 before modeling, so the measured MAE $y_{ie}$ is already in log10 units and is comparable across endpoints.

To avoid the previously shown variance inflation of the BBT model, we directly generate predictions of the population log MAE.

We assume the structure of the measured log MAE can be decomposed as:

$$
y_{ie} = \mu_e - \beta_e\theta_i + u_{ie}
$$

$\mu_e\in\mathbb{R}$ is the **endpoint difficulty**, mean log MAE across models on endpoint e.

$\theta_i \in \mathbb{R}$ is the **latent ability** of model $i$ across all endpoints. Note the negative sign in the decomposition means larger $\theta_i$ corresponds to better models.

$\beta_e \geq 0$ is the **discrimination** of endpoint $e$, the ability of endpoints to separate various models.

$u_{ie}$ is the **specialization** of model $i$ on endpoint $e$, allowing statements to be made about performance of models per endpoint. It is sampled from $\mathcal{N}(0, \omega_e^2)$ and orthogonal for differing models and endpoints. (see footnote)

Note translations across $\theta_i$ can be absorbed into the endpoint difficulty, so for identifiability, we fix $\sum_i \theta_i = 0$. Multiplication by constants can be absorbed into the discrimination, so we assign a prior to $\theta$. Simultaneous negation of $\theta_i, \beta_e$ is avoided by restricting $\beta_e$ to be non-negative.

We impose:

$$
\theta \sim \mathcal{N}(0, P), P = I_M - \frac{1}{M}\bold{11}^\top
$$

#### Priors

We use statistics of our dataset to compute hyper-parameters for priors. Define the sample variance of log base 10 MAE as:

$$
s_e^2 = \frac{1}{M-1}\sum_i (\hat{y}_{ie}-\bar{y}_e)^2
$$

and a noise-corrected median as: 

$$
s = \text{median}_e \tilde{s}_e, \tilde{s}_e^2 = \max(s_e^2 - (\bar{v}_e - \bar{c}_e), \epsilon_0)
$$

where $\bar{v}_e$ and $\bar{c}_e$ are the mean diagonal and off-diagonal of $V_e$.

$$
\mu_e \sim \mathcal{N}(\bar{y}_e, 1)\\ \beta_e = \sigma_\beta \tilde{\beta}_e, \omega_e = \sigma_\omega \tilde{\omega}_e, \\ \tilde{\beta}_e, \tilde{\omega}_e\sim\text{Half-Normal}(0, 1) \\ \sigma_\beta, \sigma_\omega \sim \text{LogNormal}(\log(s/\sqrt 2), 0.5)
$$

### Measurement

Measurement of log MAE is assumed to be observed with some noise: 

$$
\hat{y_{ie}} = y_{ie} + \epsilon_{ie}, \epsilon_{\cdot e} \sim \mathcal{N}(0, V_e)
$$

where $V_e$ is the $M \times M$ sampling covariance matrix for endpoint $e$. We also write $v_{ie} \equiv (V_e)_{ii}$ and $c_{ije} \equiv (V_e)_{ij}$.

The observation model per endpoint is an $M$-dimensional Gaussian: 

$$
\hat{y}_{\cdot e}|\theta, \mu_e, \beta_e, \omega_e \sim \mathcal{N}(\mu_e\bold{1}-\beta_e\theta, \omega^2_eI_M+V_e)
$$

We get the margin for $d_{ije}$:

$$
d_{ije} = y_{je} - y_{ie} = \beta_e(\theta_i - \theta_j) + (u_{je} - u_{ie})
$$

and conditioning on ability gives:

$$
 d_{ije} | \theta \sim \mathcal{N}(\beta_e(\theta_i - \theta_j), 2\omega_e^2)
$$

but the sampling noise makes our estimator noisy: 

$$
\hat{d}_{ije} | \theta \sim \mathcal{N}(\beta_e(\theta_i - \theta_j), s^2_{ije}) \\ s^2_{ije} = 2\omega_e^2 + v_{ie} + v_{je} - 2c_{ije}
$$

# Concrete Sketch of Steps

this is something i’m writing to not get lost in the details myself

!margin_model_data_pipeline.png

## Data Collected

Step 1:

For each endpoint, we populate a matrix $A_e \in \mathbb{R}^{+\cup 0, n_e \times M}$: 

$$
(A_e)_{ci} = |\hat{p}_{ice} - t_{ce}|
$$

the absolute error of model $i$ on compound $c$. All these numbers are already in the log-transformed space.

Step 2:

Our measured MAE (in log space) per model per endpoint is now:

$$
\hat{y}_{ie} = \frac{1}{n_e} \sum_{c=1}^{n_e} (A_e)_{ci}
$$

Step 3:

And the sampling covariance matrix, per endpoint:

$$
(S_e)_{ij} = \frac{1}{n_e-1}\sum_{c=1}^{n_e} ((A_e)_{ci} - \hat{y}_{ie})((A_e)_{cj}-\hat{y}_{je}), V_e = \frac{1}{n_e}S_e
$$

Step 4:

We need to update estimate of the following parameters:

$$
z, \text{ the free coordinates of } \theta \in \mathbb{R}^{M-1} \\ \mu_e \in \mathbb{R}^{E} \\ \tilde{\beta}_e \in \mathbb{R}^{E}_{\geq 0} \\ \tilde{\omega}_e \in \mathbb{R}^{E}_{\geq 0} \\ \sigma_\beta, \sigma_\omega \text{ each} \in \mathbb{R}
$$

Recall per endpoint $e$, our model is: 

$$
\hat{y}_{\cdot e}|\theta, \mu_e, \beta_e, \omega_e \sim \mathcal{N}(\mu_e\bold{1}-\beta_e\theta, \omega^2_eI_m+V_e)
$$

Thus, the log density is: 

$$
\ell_e = -\frac{1}{2}\log \det \Sigma_e - \frac{1}{2}r_{\cdot e}^\top \Sigma_e^{-1} r_{\cdot e}
$$

where $r_{\cdot e}$ is the residual per endpoint: 

$$
r_{\cdot e} \equiv \hat{y}_{\cdot e} - m_e = \hat{y}_e - \mu_e\bold{1} + \beta_e \theta
$$

and $\ell = \sum_e \ell_e$. We define $g_e \equiv \Sigma_e^{-1}r_{\cdot e}$.

After some work and using Jacobi’s formula, we can show:

$$
\frac{\partial \ell}{\partial z} = -\sum_e \beta_e Q^\top g_e \\ \frac{\partial \ell_e}{\partial \mu_e} = \bold{1}^\top g_e \\ \frac{\partial \ell_e}{\partial \beta_e} = -\theta^\top g_e \\ \frac{\partial \ell_e}{\partial \omega_e} = \omega_e(g_e^\top g_e - \text{tr}\Sigma_e^{-1})
$$

where $\theta = Qz$.

For numerical stability, Stan performs HMC by mapping some of these parameters to log-transformed versions. All these derivatives (including the chain rule through the log transform) are computed via Stan’s auto-differentiation.

Step 5:

Quantities to report

Once Stan finishes HMC, there will be thousands of samples for parameter estimates. We can then compute statistics from these samples, such as:

$$
\delta_{i, \text{ECFP}}^{(s)} = \theta_i^{(s)} - \theta_{\text{ECFP}}^{(s)}
$$

per sample, and then compute $P(\delta_{i,\text{ECFP}} > 0)$.

Models can be ranked in the order of $\bar{\theta}_i$, and for the sake of completion we can also compute a distribution over rank of model per draw.

Recall $\theta$ is sampled from unit variance scale, so we can convert per endpoint, the implied MAE difference is: 

$$
\Delta_{ije}^{(s)} = \beta_e^{(s)}\delta_{ij}^{(s)}
$$

then computing the 89% interval of this spread across draws gives a proper uncertainty estimate. This then allows us to pick a region of practical equivalence that is physically grounded, rather than simply based on the width of the posterior. We can state how much ‘better’ model $i$ is than model $j$ on endpoint $e$.

For parity with the original BBT paper, the question of whether model $i$ beats model $j$ on a new endpoint can be estimated by sampling new: 

$$
\beta_{\text{new}}^{(s)} \sim \text{Half-Normal}(0, \sigma_\beta^{(s)}), \omega_{\text{new}}^{(s)} \sim \text{Half-Normal}(0, \sigma_\omega^{(s)})
$$

and then we compute per draw: 

$$
P(i \succ j) = \Phi\biggl( \frac{\beta_{\text{new}}^{(s)}\delta_{ij}^{(s)}}{\sqrt2\omega_{\text{new}}^{(s)}}\biggr)
$$

and average over draws.

# Appendix

## Variance Inflation

The key issue with this model is the assumption that each pairwise trial is an independent event. At a high-level, there should only be $n$ degrees of freedom, yet this likelihood function assumes there are $\frac{n(n-1)}{2}$ degrees of freedom. 

### Computation of Score Function

Formally, assume there are $M$ models being compared across $D$ datasets, and $N_{ij} = D$. The pairwise (pseudo)log-likelihood is: 

$$
\ell_P(\beta) = \sum_{i < j} [W_{ij}\log \pi_{ij} + W_{ji}\log(1-\pi_{ij})], \pi_{ij} = \sigma(\beta_i - \beta_j) = \sigma(\eta_{ij})
$$

Recall:

$$
\text{Var}(\nabla \ell(\beta)) = -\mathbb{E}[\nabla^2 \ell]
$$

(under simple regularity conditions) for the true likelihood. 

For simplicity, assume $D = 1$ and label $X_{ij}=1$ if $i$ beats $j$. We then define $u_{ij} = \nabla_\beta \ell_{ij} \in \mathbb{R}^M$. We can chain rule to get: 

$$
u_{ij} = \frac{\partial \ell_{ij}}{\partial\pi_{ij}} \frac{\partial \pi_{ij}}{\partial\eta_{ij}}  \frac{\partial \eta_{ij}}{\partial\beta}
$$

Each factor can be computed as: 

$$
 \frac{\partial \ell_{ij}}{\partial\pi_{ij}} = \frac{X_{ij}}{\pi_{ij}} - \frac{1-X_{ij}}{1-\pi_{ij}} = \frac{X_{ij} - \pi_{ij}}{\pi_{ij}(1-\pi_{ij})} \\  \frac{\partial \pi_{ij}}{\partial\eta_{ij}} = \sigma^{'}(\eta_{ij}) = \pi_{ij}(1-\pi_{ij}) \\  \frac{\partial \eta_{ij}}{\partial\beta} = e_i - e_j
$$

where $e_i$ is the standard Cartesian basis vector, $\langle 0,0,...,0,1,0,...0,0\rangle$, a $1$ at index $i$.

Thus we get: 

$$
u_{ij} = (X_{ij} - \pi_{ij})(e_i - e_j)
$$

Note  $\mathbb{E}[u_{ij}] = (\mathbb{E}[X_{ij}] - \pi_{ij})(e_i - e_j) = 0$, which means the marginal is correct, so **the pseudo-likelihood yields a consistent estimator of $\beta$**, just the variance is inflated for finite sampling.

Define: 

$$
U \equiv \nabla_\beta \ell_P = \sum_{i<j} u_{ij}
$$

and clearly $\mathbb{E}[U] = 0$

### Fisher Information

$J \equiv \text{Var}(U) = \mathbb{E}[UU^\top]$. Expand this to get:

$$
J = \mathbb{E}\biggl[ (\sum_{i<j} u_{ij}) (\sum_{k<l} u_{kl})^\top \biggr] = \sum_{i<j}\sum_{k<l} \text{Cov}(u_{ij}, u_{kl}) \\ = \sum_{i<j}\text{Var}(u_{ij}) + \sum_{\{(i,j): i < j\} \neq \{(k, l) : k < l\}} \text{Cov}(u_{ij}, u_{kl})
$$

Now we can apply regularity conditions to write: 

$$
J = H + C
$$

where $H$ is the fisher information, the negative of the Hessian of the pseudo log likelihood, and $C$ is the covariance matrix.

However, since each sample $ij$ is not pairwise uncorrelated with other samples, we can write: 

$$
J = \text{Var}(\nabla \ell_P) = H + C \\ H = -\mathbb{E}[\nabla^2 \ell_P], C = \sum_{\{(i,j): i < j\} \neq \{(k, l) : k < l\}} \text{Cov}(u_{ij}, u_{kl})
$$

For the sake of intuition, we can also compute: 

$$
\text{Cov}(u_{ij}, u_{kl}) = \text{Cov}(X_{ij}, X_{kl}) \cdot (e_i - e_j)(e_k - e_l)^\top
$$

### Variance Inflation

Recall the actual estimator $\hat{\beta}$ will be where $U(\hat{\beta}) = 0, U(\beta) = \nabla_\beta \ell_P(\beta)$. Taylor expand around the ground truth, $\beta_0$: 

$$
0 = U(\hat{\beta}) = U(\beta_0) + \nabla_\beta U(\beta_0)(\hat{\beta} - \beta_0) + O(||\hat{\beta} - \beta_0||^2)
$$

which from our previous work gives us: 

$$
0 \approx U(\beta_0) - H(\hat{\beta} - \beta_0) \implies \hat{\beta} - \beta_0 = H^+U(\beta_0)
$$

take the variance of both sides to get:

$$
\text{Var}(\hat{\beta}) \approx H^+ \text{Var}(U(\beta_0))H^+ = H^+JH^+
$$

(note: $H$ is singular, thus we need the Moore-Penrose pseudo-inverse, $H^+$ as the naive inverse does not exist).

### Explicit Computation

Now we compute a column of $U$ as:

$$
U_i = \sum_{j\neq i}(X_{ij} - \pi_{ij}) = \sum_{j\neq i} X_{ij} - \mathbb{E}[R_i] = R_i - \mathbb{E}[R_i]
$$

where $R_i$ is the number of models that model $i$ beats other models on this dataset.

From earlier, we know the diagonal terms of $J$ are: 

$$
J_{ii} = \text{Var}(U_i) = \text{Var}(R
_i) \\
$$

and clearly $R_i = \sum_{j\neq i} X_{ij}$, so we can write: 

$$
J_{ii} = \sum_{j\neq i} \text{Var}(X_{ij}) +\sum_{j\neq k; j, k \neq i} \text{Cov}(X_{ij}, X_{ik})
$$

the first term has $M-1$ terms of $1/4$ (under uniform distribution of orderings), and the second has $(M-1)(M-2)$ terms of $1/12$ (coming from $1/3$ chance of $i$ being the best, and $1/2$ chance for each independent $i \succ j, i\succ k$). Now we can compute:

$$
\text{Var}(R_i) = \frac{M-1}{4} + \frac{(M-1)(M-2)}{12} = \frac{M^2 -1}{12}
$$

Now, the off-diagonals:

$$
J_{ij} = \text{Cov}(U_i, U_j) = \text{Cov}(R_i, R_j)
$$

Note, $\sum_i R_i = \binom{M}{2}$ is deterministic, thus: 

$$
0 = \text{Var}\biggl( \sum_{i} R_i \biggr) = M\text{Var}(R_i) + M(M-1)\text{Cov}(R_i, R_j) \\ \implies J_{ij} = -\frac{\text{Var}(R_i)}{M-1} = -\frac{M+1}{12}
$$

Now we can also compute entries of $H$, first by noting:

$$
\nabla^2_\beta \ell_{ij} = -\frac{\partial \pi_{ij}}{\partial\beta} (e_i - e_j)^\top = -\pi_{ij}(1-\pi_{ij})(e_i - e_j)(e_i - e_j)^\top
$$

and then summing over pairs: 

$$
H = \sum_{i < j} v_{ij}(e_i - e_j)(e_i - e_j)^\top
$$

We start with the diagonal entries: 

$$
H_{ii} = \sum_{j\neq i} \pi_{ij}(1-\pi_{ij}) = \sum_{j\neq i} \text{Var}(X_{ij}) = \frac{M-1}{4}
$$

and then the off-diagonal entries: 

$$
H_{ij} = -\pi_{ij}(1-\pi_{ij}) = -\text{Var}(X_{ij}) = -\frac{1}{4}
$$

We note this yields a constant scaling between $J$ and $H$:

$$
\boxed{J = \frac{M+1}{3} H}
$$

which implies: 

$$
\boxed{\text{Var}(\hat{\beta}) = H^+JH^+ = \frac{M+1}{3} H^+}
$$

Compare to the case where $C = 0$, 

$$
J = H + C = H \implies \boxed{\text{Var}(\hat{\beta}_{C=0})= H^+JH^+ = H^+}
$$

thus we clearly have variance inflation in our estimate of $\beta$ by a factor of 

$$
\boxed{\frac{M+1}{3}}
$$

Note when we add back that there are $D$ datasets, we get: 

$$
H =\sum_d H^{(d)} = DH^{(1)}, J = \text{Var}\biggl( \sum_d U^{(d)}\biggr) = DJ^{(1)}
$$

thus, 

$$
\text{Var}(\hat{\beta}) = \frac{M+1}{3D}(H^{(1)})^+, \text{Var}(\hat{\beta}_{C=0}) = \frac{1}{D}(H^{(1)})^+
$$

which means the variance inflation factor is the exact same, and unchanged by the number of datasets used.

Note: Formally, this holds for evaluation at $\beta_0 = 0$ and for each dataset producing a total ordering over the models. We also need to invoke the Bernstein-von-Mises theorem to argue that our computed variance inflation for the MLE point estimate holds for the Bayesian model.

## MCMC Methods

Often, a probability density that we want to sample from is intractable to explicitly compute for various reasons. One way around this is to construct a Markov chain whose stationary distribution is the the desired sampling distribution. Thus, the longer we simulate the Markov chain for, the closer our samples will become to the target distribution

### Metropolis-Hastings

We desire to sample from a distribution, call it $P(x)$. We want the unique stationary distribution of a constructed Markov chain to be $\pi(x)= P(x)$. 

A sufficient condition for a stationary distribution to exist is **detailed balance: $\pi(x)P(x^\prime|x) = \pi(x^\prime)P(x|x^\prime)$. Letting the stationary distribution equal $P(x)$ yields $\frac{P(x^\prime |x)}{P(x|x^\prime )} = \frac{P(x^\prime)}{P(x)}$. We write the transition probability as $P(x^\prime |x) = g(x^\prime |x)A(x^\prime ,x)$ where $g, A$ are the proposal and acceptance distributions, respectively. Inserting this relation gives $\frac{A(x^\prime , x)}{A(x, x^\prime )} = \frac{P(x^\prime )}{P(x)}\frac{g(x |x^\prime )}{g(x^\prime |x)}$. The Metropolis choice is $A(x^\prime ,x)  = \min\biggl( 1, \frac{P(x^\prime )}{P(x)}\frac{g(x |x^\prime )}{g(x^\prime |x)} \biggr)$.

Metropolis-Hastings is as follows:

1. Pick an initial state, $x_0$ at $t=0$.
2. Generate a random candidate $x^\prime$ according to $g(x^\prime | x_t)$.
3. Accept the transition with probability $A(x^\prime ,x_t)  = \min\biggl( 1, \frac{P(x^\prime )}{P(x_t)}\frac{g(x_t |x^\prime )}{g(x^\prime |x_t)} \biggr)$
4. Increment $t = t+1$.

### Hamiltonian Monte Carlo

We want to sample from a distribution, call it $P(x)$ again. Label the potential energy as $U(x) =- \log P(x)$, which can also be written as $P(x) \propto e^{-U(x)}$. We augment by introducing an auxiliary momentum to define a joint density $P(x, p) \propto \text{exp}\bigl( -U(x) - K(p)\bigr), K(p) = \frac{1}{2}p^\top M^{-1} p$ where $K(p)$ is the kinetic energy. We note the marginal of this joint distribution across $x$ produces samples from the desired distribution.

We define the Hamiltonian $H(x, p) = U(x) + K(p)$ and recall Hamilton’s equations of motion are: 

$$
\frac{dx}{dt} = \frac{\partial H}{\partial p} = M^{-1}p \text{ }\text{ and }\text{ } \frac{dp}{d_t} = -\frac{\partial H}{\partial x} = -\nabla U(x)
$$

For the sake of completion, we can prove that the Hamiltonian is conserved over time: 

$$
\frac{dH}{dt} = \nabla_x H \cdot \dot{x} + \nabla_p H \cdot \dot{p} = 0
$$

The flow is symplectic, so the phase space volume is preserved, and detailed balance is upheld by time reversibility.

Hamilton’s equations do not have a general, closed form solution. Thus the traditional method is to integrate both sides, and use a leapfrog integrator: 

$$
p_{t + \epsilon/2} =p_t - \frac{\epsilon}{2}\nabla U(x_t) \\ x_{t+\epsilon} = x_t + \epsilon M^{-1}p_{t + \epsilon /2} \\ p_{t+\epsilon} = p_{t + \epsilon /2} - \frac{\epsilon}{2}\nabla U(x_{t + \epsilon})
$$

Leapfrog integration preserves phase space volume and reversibility, but the Hamiltonian drifts as $O(\epsilon ^2)$. Thus, the Metropolis acceptance criterion is used to correct this drift. After a leapfrog step, a Metropolis-Hastings step is applied which yields: 

$$
X_{n+1}|X_{n} = x_0 = \begin{cases} x_n(L\epsilon) \text{ with probability } \alpha((x_0, p_0), (x_{L\epsilon}, p_{L\epsilon})) \\ x_0 \text{ otherwise } \end{cases}
$$

where: 

$$
\alpha((x_0, p_0), (x_{L\epsilon}, p_{L\epsilon})) = \min \biggl\{ 1, \text{exp}(H(x_0, p_0) - H(x_{L\epsilon}, p_{L\epsilon})\biggr\} 
$$

Note the momentum resampling is what allows us to move between level sets of $H$. 

Typically, if the posterior has covariance $\Sigma$, the matrix $M$ is chosen as $M = \Sigma^{-1}$, estimated during a warmup. 4 chains are also traditionally initialized to avoid posterior confinement (recall the momentum resampling is the only way we can move between level sets of $H$).