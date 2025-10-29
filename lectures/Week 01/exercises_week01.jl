### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ e9c4ec3e-7b0f-11ef-00b0-5fa0fac2ae3a
begin
    import Pkg
    Pkg.activate()
	using Makie, GLMakie, CairoMakie, WGLMakie, LaTeXStrings
	using LinearAlgebra
	using PlutoUI
end;

# ╔═╡ e6def2af-aebc-4da2-8e5c-b26b3d23eb52
begin
	using TikzPictures
	tp = TikzPicture(raw"""
		\tkzInit[xmin=-5,xmax=5,ymin=-5,ymax=5]
		
		\tkzDrawX[>=latex]
		\tkzDrawY[>=latex]
		
		\tkzDefPoint(0,0){O}
		\tkzDefPoint(4,0){A}
		
		\tkzDrawCircle(O,A)
		\tkzDefPoint(0, 3.5){La}
		\tkzDefPoint(5.5, 0){Lb}
		\tkzInterLC(La,Lb)(O,A) \tkzGetPoints{P_2}{P_1}
		\tkzDrawLine(P_1, P_2)
		\tkzLabelPoints[below left](O)
		\tkzLabelPoint[above](P_1){$P_1$}
		\tkzLabelPoint[above right](P_2){$P_2$}

		\tkzDefMidPoint(P_1,P_2)
		\tkzGetPoint{M}
		\tkzLabelPoints[above right](M)
		\tkzDrawPoints(P_1, P_2, M)

		\tkzDrawSegment(O,M)

		\tkzLabelLine[pos=0.3, above right](P_1, P_2){$\ell(\rho, \phi)$}
	""", 
		preamble=raw"""
		\usepackage{tkz-euclide}
		% \usepackage{pgfplots}
		% \usetikzlibrary {datavisualization.formats.functions} 
		\usetikzlibrary{angles,quotes}
		""",
	options = "scale=1.7")
end

# ╔═╡ 07c49846-67c5-4537-9f01-c38c8c41d4ec
begin
	struct Img
		filename
	end
	
	function Base.show(io::IO, ::MIME"image/png", w::Img)
		write(io, read(w.filename))
	end
	Img
end

# ╔═╡ de7e7217-bbb8-47e5-84bf-cf8e21896afe
begin
	global plots = Dict()
	"""
		with_backend(f, backend, args...; vargs...)
	
	Show the output figure of function `f` called with arguments `args` with the backend `backend`
	"""
	function with_backend(f, backend, args...; vargs...)
		key = hash((f, backend))
		plots[key] = haskey(plots, key) ? plots[key] + 1 : 0
		backend.activate!(inline = backend == GLMakie)
		out = f(args...; vargs...)
		if backend == GLMakie
			plots[key] != 0 && GLMakie.display(out)
			return md"Rerun to show plot!"
		else
			return out
		end
	end
end

# ╔═╡ de2d9842-cf02-4fac-9d9c-ff41529dc725
L(s) = begin @assert typeof(s) == String; return latexstring(raw"\text{"* s * "}") end

# ╔═╡ 7aef7967-38f7-4add-b92f-75c22c149f70
md"""
# Questions
1. Shouldn't ``\Omega`` be change in density or something like that in the time dimension represented by ``d``? Do I understand it correctly?
$(Img("./Q1.png"))
2. Why is ``d \neq s``? So ``d`` represents the two lateral directions and ``\Omega`` is the number of photons or something of the like, correct? Why isn't ``s = 2`` for representing both of the lateral dimensions? Are we taking the pixels sequentially by rows?
$(Img("./Q2.png"))
$(Img("./Q3.png"))
""";

# ╔═╡ 87d9265b-8302-41dd-87b1-b2d1976861bc
md"""
# Exercise 1

- __(a)__ Given a function ``x`` in ``C([0, 1])``, find a solution ``y`` in ``C^1([0, 1])`` to the equation ``y'(t) = x(t)``. Equip the equation with boundary conditions ``y(0) = 0``. Is there a solution and is it unique?
- __(b)__ Write ``y`` as an integral operator acting on ``x``. What is the forward, and what is the inverse problem?

- __(c)__ Show that the forward problem is well-posed for ``X = Y = C([0, 1])``, and for ``X = C([0, 1])``, ``Y = C^1([0, 1])``. Is the inverse problem also well-posed for both scenarios?
"""

# ╔═╡ 15df70f0-8f66-4a61-b4fb-043410d94ead
md"---"

# ╔═╡ c61ad10b-4122-4859-a63a-418a42f7a2d6
md"""
- **(a)**
Generally
```math
	y(t) \in \left\{ \int_0^t x(s)\, \mathrm{d}s + C \mid  C \in \mathbb{R} \right\}.
```
For the given boundary condition ``y(0) = 0``, we set ``C = 0``.
The solution is unique.
- **(b)**
As an integral operator, we have
```math
y(t) = \int_0^1 K(t,s) x(s) \, \mathrm{d}s.
```
If we set ``K`` to the _indicator function_ of the integration interval from **(a)**, we have the solution
```math
y(t) = \int_0^1 K(t,s) x(s) \, \mathrm{d}s = \int_0^1 \chi_{[0, t]}(s) x(s) \, \mathrm{d}s = \int_0^t x(s) \, \mathrm{d}s.
```
The forward problem is _"Given ``x`` determine ``y``"_.
The inverse problem is _"Given measurements of ``y`` detemine ``x``"_.
- **(c)** A sufficient condition for a primitive function to exist is for the function to be continuous on a superset of the integration interval. For a continuous function, its primitive function is also continuous. Thus the forward problem is well-posed for this setup.

To show that the inverse problem is not well-posed for the case of ``Y = C([0,1])`` we can show a contradiction with one of the three conditions of well-posedness...
For the scenario, where ``g \in C([0,1])``, we can take the Weirstrass function 
```math
W(x) = \sum_{n=0}^{\infty} a^n \cos(b^n \pi x)
```
``0 < a < 1`` and ``b \in \mathbb{Z}`` positive odd and ``ab > 1 + \tfrac{3}{2}\pi``, which is everywhere continuous but nowhere differentiable. This is a contradiction with existence of the inverse.
"""

# ╔═╡ fcd86291-b3eb-4f84-9216-05f3ded2da34
with_backend(WGLMakie) do
	N = 20
	a = 1/2
	b = 7 
	weirstrass(x) = sum(n -> a^n*cos(b^n*π*x), Base.OneTo(N))
	fig,ax,plt = lines(0..1, weirstrass)
	ax.title = L"\text{Weirstrass function for }a=1/2\text{ and }b = 7"
	xlims!(ax, 0,1)
	fig
end

# ╔═╡ 17bbc084-4ad9-4725-9613-cb534d189bf4
md"""
A contradiction with stability can be made by the perturbation of ``y(s)`` by ``\delta \sin(t/\delta^2)`` with the norm on continuous functions ``\|f\|_{C([a,b])} = \max_{[a,b]} |f|``:
```math
	\tilde{y}_\delta(s) = y(s) + \delta \sin (s / \delta^2),\quad \|\tilde{y} - y\| = \max_{[0,1]} |\delta \sin(s / \delta^2)| \leq \delta
```
and for the solution
```math
	\tilde{x}_\delta(s) = x(s) + \frac{1}{\delta} \cos(s / \delta^2),\quad \|\tilde{x} - x\| = \max_{[0,1]} |\frac{1}{\delta} \cos(s / \delta^2)| = 1/\delta.
```
For ``(\delta_n)``, such that ``\delta_n \to 0`` the data ``\tilde{y}_n \coloneqq \tilde{y}_{\delta_n} \to y`` with ``n \to \infty`` but ``\tilde{x}_n \coloneqq \tilde{x}_{\delta_n} \not\to x``.
"""

# ╔═╡ c567f17b-7593-458b-b318-397c2e6e64bb
with_backend(WGLMakie) do
	δ = 0.09
	y(s) = s^2*(1-s)
	y_tilde(s) = y(s) + δ*sin(s/δ^2)
	x(s) = s*(2 - 3s^2)
	x_tilde(s) = x(s) + cos(s/δ^2)/δ
	
	fig = Figure()
	ax1,_ = lines(fig[1,1],0..1, y, label = L"y")
	lines!(ax1, 0..1, y_tilde, label = L"\tilde{y}")
	ax1.title = L("Data")
	axislegend(ax1; position = :lt)
	
	ax2,_ = lines(fig[1,2],0..1, x, label = L"x")
	lines!(ax2, 0..1, x_tilde, label = L"\tilde{x}")
	ax2.title = L("Solution")
	axislegend(ax2; position = :lt)

	Label(fig[0,:], latexstring(raw"\delta = ", string(δ), raw"\quad ", "y(s) = s^2(1 -s)"))

	linkxaxes!(ax1, ax2)
	xlims!(ax2, 0,1)
	fig
end

# ╔═╡ 6f483cc6-7ec1-40ed-9dde-514813b8d75f
md"""
Using ``Y = C^1([0,1])`` solves this problem since the norm is now
```math
\|f\|_{C^1([0,1])} = \|f\|_{C([0,1])} + \|f'\|_{C([0,1])},
```
so the norm of the solution is included in the norm of the data and the any such sequence which would generate this issue diverges.
"""

# ╔═╡ b15d8136-aaba-48e5-97a4-b44884ecdfb2
md"""
For the case of ``Y = C^1([0,1])``, the problem is well-posed.
"""

# ╔═╡ 25973104-d7eb-402b-92dc-1ed66b2ed4f3
md"""
# Exercise 2

Consider the matrix

```math
A = \begin{pmatrix}
1 & 1 \\
1 & 1.001 \\
0 & 0
\end{pmatrix}
```
Consider the vectors
```math
b_1 = \begin{pmatrix}
2 \\
2 \\
0
\end{pmatrix},
b_2 = \begin{pmatrix}
2 \\
2 \\
1
\end{pmatrix},
b_3 = \begin{pmatrix}
2 \\
2.001 \\
0
\end{pmatrix}
```
For what ``j``'s is the problem ``Ax = b_j`` well-posed?
"""

# ╔═╡ b6376f35-4ea5-49aa-b689-fc600136b375
md"---"

# ╔═╡ 7499a720-e518-4e2f-b49e-2636e1b7158d
md"""
A solution exists for ``j = 1`` and ``j = 3`` and there is no other problem for the well posedness.

However the matrix is **ill-conditioned** meaning that a small error in ``b`` or ``A`` would lead to a significant difference in the solution.
This is the case for ``j = 1`` and ``j = 3`` as we can see if we solve the problem. The solution is
```math
x_1 = \begin{pmatrix}2\\0\end{pmatrix} 
\quad\text{ and }\quad
x_3 = \begin{pmatrix}1\\1\end{pmatrix},
```
which is a big difference considering the difference in ``b_1`` and ``b_3``.
"""

# ╔═╡ 67da0465-b7e4-4471-a4f4-bffc6ea9ff9f
md"""
# Exercise 3
In the Hilbert space ``L^2(0, \pi)`` the functions ``\phi_n(t) = \sqrt{\frac{2}{\pi}} \sin(nt), n = 1, 2, \ldots``, constitute an orthonormal basis. Let's define the operator ``K_N \in B(L^2(0, \pi)) (N > 1 \text{ is fixed})`` acting on ``f(t) = \sum_{n=1}^{\infty} c_n\phi_n(t)`` by
```math
K_N f(t) = \sum_{n=1}^N \frac{c_n}{n} \phi_n(t)
```
Let ``g \in L^2(0, \pi)`` be our data/measurement and consider the inverse problem of solving ``Kf = g``. Is the problem well-posed?

!!! hint
	You might want to consider ``g = \phi_M \text{ for } M > N``.
"""

# ╔═╡ ee7acd00-ca76-4f74-84b6-d385e89c0555
md"---"

# ╔═╡ a73d2009-e374-41d4-b3da-d85af5165e02
md"""
For ``g = \phi_M`` for ``M > N`` no solution ``f`` exists. We can show that by realizing that any ``Kf`` contains at most the first ``N`` basis functions in its expansion, so ``g`` is orthogonal to any such ``Kf``. 

Another problem is, that  if we have ``g(t) = \sum_{n=1}^N \tilde{c}_n \phi_n(t)``, the solution to this problem is
```math
	f(t) \in \left\{\sum_{n=1}^N (n\cdot \tilde{c}_n) \phi_n(t) + e(t)   \mid e(t) = \sum_{n=N+1}^{+\infty} c_n\phi_n(t);\,(c_n) \subset \mathbb{R}\right\},
```
so it is not unique either.
"""

# ╔═╡ db7ef968-cf95-4447-942d-2f5248f2d959
md"""
# Exercise 4
In the spirit of the previous exercise, take ``N = \infty`` and define
```math
Kf(t) = \sum_{n=1}^{\infty} \frac{c_n}{n} \phi_n(t)
```
Show that ``K \in B(L^2(0, \pi))``; consequently the forward problem is well-posed. Is the inverse problem of solving ``Kf = g`` for some given ``g \in L^2(0, \pi)`` well-posed?
!!! hint
	Look at the stability for the inverse by considering ``g = \phi_M`` as ``M \to \infty``. Can you establish a stability bound ``\|f\|_{L^2(0,\pi)} \leq C\|g\|_{L^2(0,\pi)}`` for some ``C > 0`` independent of ``f, g``?
"""

# ╔═╡ e94240f4-2afc-4cb7-b472-975c1e842f5d
md"---"

# ╔═╡ 79c34e9f-ef99-4603-9051-513c0d2bb352
md"""
The operator is linear trivially. For the boundedness, take ``f`` such that ``\|f\|^2 = 1 = \sum_{n=1}^\infty c_n^2``. It is trivial to realize that ``\|f\|^2 = \sum_{n=1}^\infty c_n^2 \geq \|Kf\|^2 = \sum_{n=1}^\infty c_n^2/n^2``, where the Parseval identity was used. This means that the operator is bounded since it implies ``\|K\| \leq 1``. Equivalence is true for ``\|f\| = \phi_1``, hence ``\|K\| = 1`` and the operator is bounded.

Note that the inverse operator is also linear. Establishing a stability bound is impossible, since for ``g=\phi_M`` with ``M \to \infty`` the solution would be ``f = M\cdot\phi_M``. For ``\tilde{g}``, we can propose a sequence ``(\tilde{g}_n)_{n=1}^\infty`` with the the error decreasing to ``0`` but the error of the preimage is constant ``1``. This sequence is
```math
\tilde{g}_n = \tilde{g} + \frac{\phi_n}{n} \overset{{K^{-1}}}{\longmapsto} \tilde{f} + \phi_n = \tilde{f}_n.
```
This contradicts the existence of any stability bound.
"""

# ╔═╡ 6cd226e0-bf75-4c5d-b793-75fda2718fe4
md"""
# Exercise 5
Consider the operator ``K`` on ``L^2(-1, 1)`` defined by
```math
Kx(t) = \frac{1}{2t} \int_{-t}^t x(s)ds, \quad 0 < t < 1
```
and the inverse problem
```math
Kx(t) = y(t)
```
 - __(a)__ Discuss what information ``y`` carries about ``x``?
 - __(b)__ Show that ``z(t) = ty(t)`` satisfies
```math
z' = x_{\text{even}}, \quad z(0) = 0
```
where ``x_{\text{even}}`` is the even part of ``x``.
 - __(c)__ Is the inverse problem ``Kx = y`` well-posed for, say, assuming ``x`` is an even function and ``ty \in L^2(0, 1)`` or ``ty \in C^1([0, 1])``?
"""

# ╔═╡ 35b6c277-412e-40f1-8034-fb153cfbee6f
md"---"

# ╔═╡ cc6f0c15-6e2e-4ba6-8447-930fa381e310
md"""
- **(a)** ``y(t)`` is the average of ``x`` in the interval ``(-t, t)``.
- **(b)** 
```math
z(t) = ty(t) = \frac{1}{2} \bigg(\underbrace{\int_{-t}^{t} x_\text{odd}(s) \,\mathrm{d}s}_{=0} + \int_{-t}^{t} x_\text{even}(s) \,\mathrm{d}s\bigg) = \frac{1}{2}\int_{-t}^{t} x_\text{even}(s) \,\mathrm{d}s = \int_{0}^{t} x_\text{even}(s) \, \mathrm{d}s
```
which is the primitive function for ``x_\text{even}``. For ``z(0)`` the integral is taken over a one element null sized set and is therefore equal to ``0``.
- **(c)** This is the same as the derivative problem divided by ``t`` from exercise ``1``. For ``ty \in L^2(0,1)`` the function doesn't necessarily need to be differentiable. As an example, we can take again the Weirstrass function. For ``ty \in C^1([0,1])`` the inverse problem is well-posed. The null space is the linear span of all odd functions.
"""

# ╔═╡ 5e38fbdf-eeac-4108-97f8-0ad99dbb7d3c
md"""
# Exercise 6
Consider the forward heat equation taking initial condition ``f \in L^2(0, \pi)`` to terminal condition ``g \in L^2(0, \pi)``. Show that the problem has a unique solution. Establish using Fourier series and Parseval's identity the stability estimate ``\|g\|_{L^2(0,\pi)} \leq \|f\|_{L^2(0,\pi)}``.
"""

# ╔═╡ 21d68081-8e51-4f15-88b6-d6253ffd471b
md"---"

# ╔═╡ c2d63cda-c9ed-427e-9713-530d43bde57e
md"""
```math
g(x) = u(x,T) = \sqrt{\frac{2}{\pi}} \sum_{n=1}^{\infty} c_n e^{-Tn^2} \sin(nx).
``` where
```math
f(x) = \sqrt{\frac{2}{\pi}} \sum_{n=1}^{\infty} c_n \sin(nx),
``` and
```math
c_n = \sqrt{\frac{2}{\pi}} \int_0^{\pi} f(y) \sin(ny)\,\mathrm{d}y.
```
By the Parseval identity using the previous expansions, we have  
```math
\|g\|^2  = \sqrt{\frac{\pi}{2}} \sum_{n=1}^\infty \left|c_n e^{-Tn^2}\right|^2\leq \sqrt{\frac{\pi}{2}}\sum_{n=1}^\infty |c_n|^2 = \|f\|^2,
``` where the inequality is true by elements because ``T > 0``.
"""

# ╔═╡ 032e99b2-69a0-420d-8fe0-845605287475
md"""
# Exercise 7
Consider the backwards heat equation defined for functions ``g`` in the range of the forward operator. Is the solution unique? Can you establish a stability bound ``\|f\|_{L^2(0,\pi)} \leq C\|g\|_{L^2(0,\pi)}`` for some ``C > 0`` independent of ``f, g``?
"""

# ╔═╡ 2b2e09d8-21bb-46a2-adb5-2bbdffeb0025
md"---"

# ╔═╡ aed9614f-b20f-4bd3-b35b-37daed365811
md"""
The solution is unique since a expansion in a orthonormal basis uniquely determines the function and
```math
	g(t) = \sqrt{\pi/2}\sum_{n = 1}^\infty c_n e^{-Tn^2} \sin(nt) = \sqrt{\pi/2}\sum_{n=1}^\infty d_n \sin(nt)
``` implies
```math
	f(t) = \sqrt{\pi/2}\sum_{n = 1}^\infty \underbrace{\frac{d_n}{e^{-Tn^2}}}_{c_n} \sin(nt).
```
It is impossible to establish a stability bound for the problem. We can prove this adding a small perturbation to ``g`` and observing the change in ``f``. Notice that
```math
\frac{1}{e^{-Tn^2}} \to \infty \quad\text{for}\quad n \to \infty.
```
Now for the parametrized perturbation of the data we can define
```math
	\tilde{g}_{N,\alpha} \overset{!}{=} g + \alpha\sqrt{\pi/2}\,\sin(Nt),
``` for which the triangular inequality gives
```math
\| \tilde{g}_{N, \alpha} \| \leq \|g\| + \alpha \left\| \sqrt{\pi / 2 } \sin(Nt) \right\| = \|g\| + \alpha
```
and for the solution by the alternative triangular inequality, we have
```math
\|\tilde{f}_{N, \alpha}\| \geq \left| \|f\| - \left\| -\frac{\alpha}{e^{-TN^2}} \sqrt{\pi/2} \sin(Nt)\right\| \right| =  \left| \|f\| - \frac{\alpha}{e^{-TN^2}} \right| \to \infty \quad \text{for} \quad N \to \infty,
```
hence for any ``\alpha, \beta \in \mathbb{R}^+``, we can choose a pertubation (choice of ``N``) such that the error in the data is less than ``\alpha`` but the error in the solution is more that ``\beta``. This contradicts the existence of any stability bound.
"""

# ╔═╡ 9f54e8a6-6a96-4cdc-8b2e-45393401cef2
md"""
# Exercise 8
Let ``B`` be the unit ball in the plane and suppose that ``f \in C(\mathbb{R}^2)`` is a radially symmetric function with support inside ``B`` (i.e. ``f(x, y) = 0`` for ``x^2 + y^2 \geq 1``.) Show that the Radon transform
```math
R(\rho, \phi) = \int_{l(\rho,\phi)} f(x, y) ds
```
is independent of ``\phi``.
"""

# ╔═╡ eecb4c8d-9f48-438d-a5d0-896de865e95a
md"---"

# ╔═╡ 08cf6ca5-6d1d-4d82-9603-087759466110
md"""
We parametrize the line as
```math
	\ell(\rho, \phi) = (O_x + \Delta x, O_y + \Delta y)
```
where ``O_x`` and ``O_y`` are the coordinates of the midpoint of the secant line of the support circle ``x^2 + y^2 = 1`` and the line ``\ell(\rho, \phi)`` and ``\Delta x = \Delta x(s)``, ``\Delta y = \Delta y (s)`` are the offsets from the midpoint ``(O_x, O_y)`` expressed in the terms of the distance ``s`` along the line from the midpoint. If parametrize these points using ``\rho`` and ``\phi``, we get
```math
	\ell(\rho, \phi) = (\overbrace{\underbrace{\rho \cos (\phi)}_{O_x} + \underbrace{s \sin(\phi)}_{\Delta x}}^{\ell_x}, \overbrace{\underbrace{\rho \sin(\phi)}_{O_y} \underbrace{- r \cos(\phi)}_{\Delta y}}^{\ell_y}).
```
Now we can write the definition integral as
```math
	R(\rho, \phi) = \int_{-\sqrt{1 - \rho^2}}^{\sqrt{1 - \rho^2}} f(\ell_x(s), \ell_y(s)) \,\mathrm{d} s.
```
Using the symmetry of ``f``, we can rewrite this as
```math
	R(\rho, \phi) = \int_{-\sqrt{1 - \rho^2}}^{\sqrt{1 - \rho^2}} \tilde{f}\big(\underbrace{\ell_x(s)^2 +\ell_y(s)^2}_{\psi^2}\big) \,\mathrm{d} s.
```
If we expand ``\psi^2``, we get
```math
	\psi^2 = (\rho \cos (\phi) + s \sin(\phi))^2 + (\rho \sin(\phi) - r \cos(\phi))^2 = \rho^2 + s^2
```
which does not depend on ``\phi`` therefore we have the independece of the transform on ``\phi`` since it appears nowhere else in the integral.
"""

# ╔═╡ Cell order:
# ╠═e9c4ec3e-7b0f-11ef-00b0-5fa0fac2ae3a
# ╟─07c49846-67c5-4537-9f01-c38c8c41d4ec
# ╟─de7e7217-bbb8-47e5-84bf-cf8e21896afe
# ╟─de2d9842-cf02-4fac-9d9c-ff41529dc725
# ╟─7aef7967-38f7-4add-b92f-75c22c149f70
# ╟─87d9265b-8302-41dd-87b1-b2d1976861bc
# ╟─15df70f0-8f66-4a61-b4fb-043410d94ead
# ╟─c61ad10b-4122-4859-a63a-418a42f7a2d6
# ╟─fcd86291-b3eb-4f84-9216-05f3ded2da34
# ╟─17bbc084-4ad9-4725-9613-cb534d189bf4
# ╟─c567f17b-7593-458b-b318-397c2e6e64bb
# ╟─6f483cc6-7ec1-40ed-9dde-514813b8d75f
# ╟─b15d8136-aaba-48e5-97a4-b44884ecdfb2
# ╟─25973104-d7eb-402b-92dc-1ed66b2ed4f3
# ╟─b6376f35-4ea5-49aa-b689-fc600136b375
# ╟─7499a720-e518-4e2f-b49e-2636e1b7158d
# ╟─67da0465-b7e4-4471-a4f4-bffc6ea9ff9f
# ╟─ee7acd00-ca76-4f74-84b6-d385e89c0555
# ╟─a73d2009-e374-41d4-b3da-d85af5165e02
# ╟─db7ef968-cf95-4447-942d-2f5248f2d959
# ╟─e94240f4-2afc-4cb7-b472-975c1e842f5d
# ╟─79c34e9f-ef99-4603-9051-513c0d2bb352
# ╟─6cd226e0-bf75-4c5d-b793-75fda2718fe4
# ╟─35b6c277-412e-40f1-8034-fb153cfbee6f
# ╟─cc6f0c15-6e2e-4ba6-8447-930fa381e310
# ╟─5e38fbdf-eeac-4108-97f8-0ad99dbb7d3c
# ╟─21d68081-8e51-4f15-88b6-d6253ffd471b
# ╟─c2d63cda-c9ed-427e-9713-530d43bde57e
# ╟─032e99b2-69a0-420d-8fe0-845605287475
# ╟─2b2e09d8-21bb-46a2-adb5-2bbdffeb0025
# ╟─aed9614f-b20f-4bd3-b35b-37daed365811
# ╟─9f54e8a6-6a96-4cdc-8b2e-45393401cef2
# ╟─eecb4c8d-9f48-438d-a5d0-896de865e95a
# ╟─08cf6ca5-6d1d-4d82-9603-087759466110
# ╟─e6def2af-aebc-4da2-8e5c-b26b3d23eb52
