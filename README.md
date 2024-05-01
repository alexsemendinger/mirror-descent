# Mirror Descent Experiments




### Background  
[Stochastic Gradient / Mirror Descent: Minimax Optimality and Implicit Regularization](https://arxiv.org/pdf/1806.00952.pdf)

[Data-Driven Mirror Descent in Input-Convex Neural Networks](https://arxiv.org/pdf/2206.06733.pdf)

### Question

Above tells us that mirror descent on data $\{x_i, y_i\}$ with potential $\phi$ converges to $\operatorname*{argmin}_{w \in W^*} D_{\phi} (w \| w_0)$ (where $W^*$ is the set of weights that interpolate the data and $w_0$ is the intialization).

**Can we learn a useful data-dependent $\phi$ that improves generalization?**
