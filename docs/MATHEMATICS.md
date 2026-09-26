# Mathematical notes

The baseline solves
[
min_x g(x)+h(Kx),
]
with block-separable (g(x)=sum_i g_i(x_i)).

It performs a full dual proximal step and one randomized primal block update per iteration. Default step sizes satisfy
[
	ausigma|K|_2^2<1.
]

For the LASSO example,
[
min_x rac12|Kx-b|_2^2+lambda|x|_1,
]
the primal proximal map is soft-thresholding. With (h(z)=rac12|z-b|^2),
[
h^*(y)=rac12|y|^2+langle y,bangle,qquad
operatorname{prox}_{sigma h^*}(y)=rac{y-sigma b}{1+sigma}.
]

The research branches explore adaptive sampling and batched block updates. Those variants can require stronger convergence assumptions than the baseline and are kept isolated until validated.
