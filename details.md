For a **joint ARMA–GARCH fit**, robust SE are obtained exactly the same way as for a “plain” GARCH QMLE, except the parameter vector includes **both** the ARMA mean parameters and the GARCH variance parameters.

Write the full model as, for example,

[
y_t = \mu_t(\theta_m) + \varepsilon_t,
\qquad
\varepsilon_t = \sqrt{h_t(\theta_m,\theta_v)} z_t,
]

with

[
h_t
===

\omega
+
\sum_{j=1}^q \alpha_j \varepsilon_{t-j}^2
+
\sum_{j=1}^p \beta_j h_{t-j}.
]

The joint parameter vector is

[
\theta =
(\theta_m', \theta_v')',
]

where (\theta_m) contains the ARMA parameters and (\theta_v) contains the GARCH parameters.

Under Gaussian QMLE, the per-period quasi-log-likelihood is usually

[
\ell_t(\theta)
==============

-\frac12
\left[
\log h_t(\theta)
+
\frac{\varepsilon_t(\theta)^2}{h_t(\theta)}
\right],
]

up to constants. Then you maximize

[
\hat\theta
==========

\arg\max_\theta
\sum_{t=1}^T \ell_t(\theta).
]

The robust covariance matrix is the sandwich covariance for the **entire joint estimator**:

[
\widehat{\operatorname{Var}}(\hat\theta)
========================================

\frac{1}{T}
\hat H^{-1}
\hat J
\hat H^{-1},
]

where

[
\hat H
======

-\frac{1}{T}
\sum_{t=1}^T
\frac{\partial^2 \ell_t(\hat\theta)}
{\partial \theta \partial \theta'}
]

and

[
\hat J
======

\frac{1}{T}
\sum_{t=1}^T
s_t(\hat\theta)s_t(\hat\theta)',
\qquad
s_t(\hat\theta)
===============

\frac{\partial \ell_t(\hat\theta)}
{\partial \theta}.
]

So the answer is: **compute the score and Hessian with respect to the full ARMA–GARCH parameter vector, not separately for the ARMA part and the GARCH part.**

This is the Bollerslev–Wooldridge robust covariance logic for QMLE in dynamic models with time-varying covariances; it is the standard “robust” covariance used in GARCH packages. ([Duke Economics][1])

The important point is that the score vector is joint:

[
s_t(\hat\theta)
===============

\begin{bmatrix}
s_{t,m}(\hat\theta) \
s_{t,v}(\hat\theta)
\end{bmatrix},
]

so the sandwich covariance has the block structure

[
\widehat{\operatorname{Var}}(\hat\theta)
========================================

\begin{bmatrix}
\widehat V_{m,m} & \widehat V_{m,v} \
\widehat V_{v,m} & \widehat V_{v,v}
\end{bmatrix}.
]

The off-diagonal blocks matter. They capture estimation interaction between the conditional mean and conditional variance parameters. For example, uncertainty in the ARMA parameters affects the residuals (\varepsilon_t(\hat\theta)), and those residuals enter the GARCH recursion. So for a joint fit, you should **not** treat the ARMA residuals as fixed when computing GARCH SE.

For inference on only the GARCH parameters, take the relevant submatrix:

[
\widehat{\operatorname{Var}}(\hat\theta_v)
==========================================

\widehat V_{v,v}.
]

For inference on only the ARMA parameters, take

[
\widehat{\operatorname{Var}}(\hat\theta_m)
==========================================

\widehat V_{m,m}.
]

The robust standard errors are then

[
\operatorname{se}(\hat\theta_j)
===============================

\sqrt{
\left[
\widehat{\operatorname{Var}}(\hat\theta)
\right]_{j,j}
}.
]

A subtle but important distinction: in the standard GARCH QMLE setting, (\hat J) is usually the **outer product of contemporaneous scores**, not a Newey–West HAC matrix. The reason is that under correct conditional mean and variance specification, the score is a martingale difference sequence. If you want robustness to more general serial dependence in the score, then you would replace (\hat J) by a HAC long-run covariance estimate,

[
\hat J_{\text{HAC}}
===================

\sum_{k=-L}^{L}
w_k
\hat\Gamma_k,
]

but that is stronger than the usual Bollerslev–Wooldridge GARCH robust SE.

So, operationally:

1. Fit the ARMA–GARCH by one likelihood.
2. At the optimum (\hat\theta), compute (\ell_t(\hat\theta)) for each (t).
3. Compute the per-observation score (s_t(\hat\theta)) with respect to **all** parameters.
4. Compute the Hessian of the full log-likelihood with respect to **all** parameters.
5. Form the sandwich covariance.
6. Extract the diagonal or desired sub-blocks.

The two-stage case is different. If you fit ARMA first and then GARCH on residuals, the second-stage GARCH SE are generally not the same as the joint QMLE SE unless you correct for generated-regressor / first-stage estimation effects. But in the joint likelihood case, that correction is already built into the full sandwich covariance because the ARMA and GARCH parameters are estimated together.

[1]: https://public.econ.duke.edu/~boller/Econ.350/bw_1992.pdf?utm_source=chatgpt.com "quasi-maximum likelihood estimation and inference in"

Yes. You can reuse a lot of the GARCH derivatives, but you need to **augment them**, not simply paste the old covariance block into a larger matrix.

The key issue is this:

In a standalone GARCH fit, you usually treat residuals (\varepsilon_t) as data.
In a joint ARMA–GARCH fit, residuals are functions of the ARMA parameters:

[
\varepsilon_t = y_t - \mu_t(\theta_m).
]

So the variance recursion also depends on the ARMA parameters indirectly:

[
h_t = h_t(\varepsilon_{t-1}, \varepsilon_{t-2}, \ldots; \theta_v)
= h_t(\theta_m, \theta_v).
]

Therefore you need derivatives of (h_t) with respect to **all** parameters, not just the GARCH parameters.

For a Gaussian-type likelihood, or more generally a standardized density likelihood, write

[
z_t = \frac{\varepsilon_t}{\sqrt{h_t}},
]

and

[
\ell_t(\theta)
==============

## \log f(z_t; \eta)

\frac12 \log h_t.
]

Then the score for any parameter (\theta_k) can be written by chain rule as

[
s_{t,k}
=======

\frac{\partial \ell_t}{\partial \varepsilon_t}
\frac{\partial \varepsilon_t}{\partial \theta_k}
+
\frac{\partial \ell_t}{\partial h_t}
\frac{\partial h_t}{\partial \theta_k}
+
\frac{\partial \ell_t}{\partial \theta_k}\Bigg|_{\text{direct}}.
]

That decomposition is the useful part. Your existing GARCH code probably already gives you pieces like

[
\frac{\partial \ell_t}{\partial h_t},
\qquad
\frac{\partial h_t}{\partial \theta_v},
\qquad
\frac{\partial \ell_t}{\partial \eta},
]

depending on the density. You can reuse those. What you need to add is

[
\frac{\partial \varepsilon_t}{\partial \theta_m}
]

and

[
\frac{\partial h_t}{\partial \theta_m}.
]

For a standard GARCH((p,q)),

[
h_t
===

\omega
+
\sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2
+
\sum_{j=1}^p \beta_j h_{t-j}.
]

The derivative recursion for any parameter (\theta_k) is

[
\frac{\partial h_t}{\partial \theta_k}
======================================

\mathbf 1_{{\theta_k=\omega}}
+
\sum_{i=1}^q
\mathbf 1_{{\theta_k=\alpha_i}}
\varepsilon_{t-i}^2
+
\sum_{i=1}^q
2\alpha_i \varepsilon_{t-i}
\frac{\partial \varepsilon_{t-i}}{\partial \theta_k}
+
\sum_{j=1}^p
\mathbf 1_{{\theta_k=\beta_j}}
h_{t-j}
+
\sum_{j=1}^p
\beta_j
\frac{\partial h_{t-j}}{\partial \theta_k}.
]

For GARCH parameters, usually

[
\frac{\partial \varepsilon_t}{\partial \theta_v}=0,
]

so this collapses to the familiar GARCH derivative recursion.

But for ARMA parameters,

[
\frac{\partial \varepsilon_t}{\partial \theta_m}\neq 0,
]

so the term

[
\sum_{i=1}^q
2\alpha_i \varepsilon_{t-i}
\frac{\partial \varepsilon_{t-i}}{\partial \theta_m}
]

is what you are missing if you treat the residuals as fixed.

For an ARMA model written as

[
y_t
===

c
+
\sum_{i=1}^r \phi_i y_{t-i}
+
\varepsilon_t
+
\sum_{j=1}^s \vartheta_j \varepsilon_{t-j},
]

the residual recursion is

[
\varepsilon_t
=============

## y_t

## c

## \sum_{i=1}^r \phi_i y_{t-i}

\sum_{j=1}^s \vartheta_j \varepsilon_{t-j}.
]

Then its parameter derivatives are recursive too. For example,

[
\frac{\partial \varepsilon_t}{\partial c}
=========================================

## -1

\sum_{j=1}^s
\vartheta_j
\frac{\partial \varepsilon_{t-j}}{\partial c},
]

[
\frac{\partial \varepsilon_t}{\partial \phi_i}
==============================================

## -y_{t-i}

\sum_{j=1}^s
\vartheta_j
\frac{\partial \varepsilon_{t-j}}{\partial \phi_i},
]

and

[
\frac{\partial \varepsilon_t}{\partial \vartheta_j}
===================================================

## -\varepsilon_{t-j}

\sum_{\ell=1}^s
\vartheta_\ell
\frac{\partial \varepsilon_{t-\ell}}{\partial \vartheta_j}.
]

The sign convention depends on how you define the ARMA model, but structurally this is what you need.

So the answer is:

**You do not need to write an entirely new covariance estimator.**
You do need to build a **joint score matrix**.

The score matrix should have rows

[
s_t(\hat\theta)
===============

\begin{bmatrix}
s_{t,m}(\hat\theta) \
s_{t,v}(\hat\theta) \
s_{t,\eta}(\hat\theta)
\end{bmatrix},
]

where (\theta_m) are ARMA parameters, (\theta_v) are volatility parameters, and (\eta) are density parameters such as Student-(t) degrees of freedom or skewness parameters.

Then form

[
\hat J
======

\frac{1}{T}
\sum_{t=1}^T
s_t(\hat\theta)s_t(\hat\theta)'.
]

The part that cannot be obtained from a standalone GARCH covariance routine is the **cross-information** between mean and variance parameters. That comes automatically if you compute the full score vector observation by observation.

For the Hessian part,

[
\hat H
======

-\frac{1}{T}
\sum_{t=1}^T
\frac{\partial^2 \ell_t(\hat\theta)}
{\partial \theta \partial \theta'},
]

you have a few options:

1. **Fully analytic Hessian**: fastest and cleanest, but most work.
2. **Numerical Hessian of the full likelihood**: often acceptable and much easier.
3. **Numerical Jacobian of the full score**: usually better than finite-differencing the scalar likelihood.
4. **Expected/BHHH-style approximation**: sometimes used, but for robust QMLE the safest general sandwich is still based on a proper full (\hat H) and full (\hat J).

The robust covariance is then

[
\widehat{\operatorname{Var}}(\hat\theta)
========================================

\frac{1}{T}
\hat H^{-1}
\hat J
\hat H^{-1}.
]

The practical coding architecture I would use is:

```text
given theta:
    compute eps_t(theta_m)
    compute d_eps_t / d_theta_m

    compute h_t(theta_m, theta_v)
    compute d_h_t / d_all_theta

    compute ell_t
    compute d_ell_t / d_eps_t
    compute d_ell_t / d_h_t
    compute direct density scores

    assemble full score s_t
```

Then your old GARCH derivative functions can be reused for:

```text
d_h_t / d_theta_v
d_ell_t / d_h_t
density parameter scores
possibly d_ell_t / d_eps_t
```

But they must be generalized so that (h_t) can also be differentiated with respect to the ARMA parameters through (\varepsilon_t).

The main thing not to do is this:

[
\widehat V_{\text{joint}}
\neq
\begin{bmatrix}
\widehat V_{\text{ARMA}} & 0 \
0 & \widehat V_{\text{GARCH}}
\end{bmatrix}.
]

That would ignore the fact that ARMA parameters affect the volatility recursion through the residuals, and it would also miss the covariance between mean and variance estimates.
