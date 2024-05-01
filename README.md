# Mirror Descent Experiments

### Background  
[Stochastic Gradient / Mirror Descent: Minimax Optimality and Implicit Regularization](https://arxiv.org/pdf/1806.00952.pdf)

[Data-Driven Mirror Descent in Input-Convex Neural Networks](https://arxiv.org/pdf/2206.06733.pdf)

### Question

Above tells us that mirror descent on data $\{x_i, y_i\}$ with potential $\phi$ converges to $\operatorname*{argmin}_{w \in W^*} D_{\phi} (w \| w_0)$ (where $W^*$ is the set of weights that interpolate the data and $w_0$ is the intialization).

**Can we learn a useful data-dependent $\phi$ that improves generalization?**

### Roadmap

* Linear model: $y_i = w_*^\top x_i + \varepsilon_i$, potential $\phi(x) = x^\top Q x$
    * Verify that you learn a $Q$ with good val loss
    * How does learned $Q$ relate to $w_*$?
        * Can you write a closed-form expression for minimizer? How close do we get?
    * If you resample $w_* \sim \mathcal N(0, \Sigma)$, do you learn $\Sigma$?
    * Different loss functions (inner & outer loops; cross-validation)

* Linear model, more general choices of $\phi$
    * Exponential weights?
    * Input-convex neural network?
    * Non-convex "potentials"? (Do they break immediately? Do they tend to learn convex-ish things? Do they work better for some reason?)

* More realistic data (e.g. MNIST and scale up), general $\phi$ (ICNN?)