### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 5c07f4cc-71a5-11ef-3aec-efbcf53d4600
begin
    import Pkg
    Pkg.activate()
	using GLMakie, Makie, CairoMakie, WGLMakie, LaTeXStrings
	using LinearAlgebra
	using PlutoUI
end;

# ╔═╡ 85fd322e-aaad-43d0-8c48-411108ffe108
using Interpolations

# ╔═╡ b5df0251-18ab-4e70-b087-4d245917e7ce
using FFTW

# ╔═╡ a5c10b34-1424-4f63-bd1c-7fb35367cfba
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

# ╔═╡ 0ec62ab4-d32a-4488-b132-f4edcd642095
L(s) = begin @assert typeof(s) == String; return latexstring(raw"\text{"* s * "}") end

# ╔═╡ 621d8e84-9f44-4683-8c79-26558ad94efe
md"""
# 1. Warm up
The numerical differentiation problem is an inverse problem (see slides from the morning
in week 1). The problem is implemented in the code `NumericalDifferentiation.m`
to be found in learn (see the module Matlab). Try out the code.
- Change the function to ``y(s) = s^2e^{−s}``.
- Solve the inverse problem and compare to the true solution.
- Experiment with the noise level and the number of discretization points.
"""

# ╔═╡ fab1b14a-57bb-4cc1-8e61-dfa9448f235d
md"---"

# ╔═╡ 388a8ac5-6932-4497-a482-7d8ebdc647f9
begin
	y(s) = s^2*exp(-s)
	x(s) = exp(-s)*(2s - s^2)
end;

# ╔═╡ 99c02d8a-93f5-4ad1-94d3-40cfafaa25bc
begin
	N_1 = 500
	h = 1/N_1
	s = range(0, 1-h, step=h)
end;

# ╔═╡ 2fb5a6da-18dc-4053-b592-c6ae2f2d5a32
begin
	xs = x.(s)
	A = tril(ones(N_1,N_1)).*h
	ys = A * xs
end;

# ╔═╡ 51daabce-a894-4bd7-9be0-b23b29b067fb
with_backend(WGLMakie) do
	f,ax,_ = lines(s,xs; label=L"x(s)")
	lines!(s .+ h, ys; label=L"y_\text{est}(s)")
	lines!(s .+ h, sph -> y(sph); label = L"y(s)")
	ax.title = L("Forward problem")
	axislegend(ax; position = :lt)
	# xlims!(ax, 0,1)
	return f
end

# ╔═╡ 9cfedb8e-3d10-48b6-9544-e52bd4a67ef5
with_backend(WGLMakie) do
	f,ax,_ = lines(s,xs; label=L"x(s)")
	slg = SliderGrid(f[2,1], (label = L"\rho", range = range(1e-6,1e-4, 100), startvalue = 0.5e-5))
	ρ = slg.sliders[1].value
	random = randn(N_1)
	noise = @lift $ρ * random 
	ys_noisy = @lift ys .+ $noise
	xs_rec = @lift A \ $ys_noisy
	err_rel = @lift norm(xs .- $xs_rec)/norm(xs)

	Label(f[0,:], @lift(latexstring(raw"\text{Derivative reconstruction with relative noise }", round($err_rel, sigdigits = 3))), tellwidth = false)
	
	lines!(s, xs_rec; label=L"x_\text{reconstructed}(s)")
	lines!(s .+ h, ys_noisy; label=L"y_\text{noisy}(s)")
	lines!(s .+ h, sph -> y(sph); label = L"y(s)")
	axislegend(ax; position = :lt)
	# xlims!(ax, 0,1)
	f
end

# ╔═╡ b7c06aed-0df0-4b58-aa9b-6069f66700fc
md"""
# 2. Abel operator by quadrature and collocation (Nystrom method)

The operator $K$ is defined as:

$$Kx(t) = \frac{1}{\sqrt{\pi}} \int_0^t \frac{x(s)}{\sqrt{t-s}} \mathrm{d}s$$

$$= \int_0^1 k(t,s)x(s) \mathrm{d}s \quad \text{with} \quad k(t,s) = \begin{cases}
\frac{1}{\sqrt{\pi(t-s)}}, & s < t \\
0, & s > t
\end{cases}$$

is approximated by a matrix $A$ as follows: Take $h = 1/N$ and define
- ``s``-grid: $s_1 = h/2, s_2 = h/2 + h, \ldots, s_N = h/2 + (N-1)h$
- ``t``-grid: $t_1 = h, t_2 = 2, \ldots, t_N = Nh = 1$.

Then set $A_{ij} = hk(t_i, s_j)$, i.e. apply

$$Kf(t_i) \approx h \sum_j k(t_i, s_j)f(s_j).$$

- Implement the method numerically to get the matrix $A$. Experiment with different functions $x$.

- Logplot the singular values of the $A$. (Use `[U,S,V] = svd(A)` in MATLAB.)

- Try having similar $t$ and $s$ grids. Find a way for handling the singularity.
"""

# ╔═╡ cde267d6-e522-414e-9db2-799dcd98782d
begin
	N2 = 1000
	h2 = 1/N2
	s_grid = -h2/2 .+ (1:N2) .* h2
	t_grid = h2 .* (1:N2)
end;

# ╔═╡ a7986dc1-7b1e-40d5-a8ec-7d920f3c84eb
k(t,s) = s > t ? 0 : 1/sqrt(π*(t-s))

# ╔═╡ 5e5b4ea2-d4db-412a-9498-7dc1e16afef1
A_mat(s_grid, t_grid) = h2 .* [k(t_i,s_j) for t_i in t_grid,  s_j in s_grid];

# ╔═╡ de08d0cc-4291-4bd2-a23a-746e989b694e
x_funcs = [s -> s^2*exp(-s), s -> s*exp(-s)]

# ╔═╡ 8940d452-c226-4035-8309-e719b6293d7e
Kf_rec(f, s_grid, t_grid, A=A_mat(s_grid, t_grid)) = h2 .* A * f.(s_grid)

# ╔═╡ c3ecff37-af7b-48cd-9892-33c1864e6738
md"### For distinct ``s`` and ``t`` grids"

# ╔═╡ 40b721e1-9f10-44c3-85e1-e3174c7f083e
with_backend(WGLMakie) do
	fig,ax,_ = lines(t_grid, Kf_rec(x_funcs[1], s_grid, t_grid); label = L"K\,[s^2 \exp(-\!s)]")
	_ = lines!(ax, t_grid, Kf_rec(x_funcs[2], s_grid, t_grid); label = L"K\,[s\, \exp(-\!s)]")
	# lines!(t_grid, f.(t_grid))
	axislegend(ax; position=:lt)
	# xlims!(ax, 0,1)
	ax.title = L("Forward problem solutions")
	fig
end

# ╔═╡ ae6019a8-8a58-4ece-b373-822203b5664a
U,S,V = svd(A_mat(s_grid, t_grid));

# ╔═╡ 51245408-d932-49b6-b444-466518e8e0ca
with_backend(WGLMakie) do
	fig = Figure()
	ax = Axis(fig[1,1], yscale = log10)
	scatter!(ax, 1:length(S), S)
	# xlims!(ax, 0, length(S))
	ax.title = L("Logplot of the singular values")
	fig
end

# ╔═╡ f02ef0b6-28e2-413a-9168-038ed5dfbb18
minimum(S)

# ╔═╡ a72c1bd1-7210-428a-b190-90656db1d001
md"### For similar ``s`` and ``t`` grids"

# ╔═╡ 8062f5b8-261d-41a1-959d-a5f34d154db1
begin
	t_grid2 = h2 .* (1:N2)
	s_grid2 = t_grid2
end;

# ╔═╡ 0dad41e6-0bfa-4987-a73f-672a42eb31a8
md"We mask the values, where the sigularities occur. I.e. the indices where ``t_i \approx s_j``."

# ╔═╡ 8d77a0f8-ada4-4181-91c9-318103c741e1
begin
	A_mat_similar = A_mat(s_grid2, t_grid2)
	for i in axes(A_mat_similar, 1)
		A_mat_similar[i,i] = 0 # masking the values that are singular 
	end
end

# ╔═╡ 6c7c602b-8cf6-4f2e-b544-faa94f0146db
with_backend(WGLMakie) do
	fig,ax,_ = lines(t_grid2, Kf_rec(x_funcs[1], s_grid2, t_grid2, A_mat_similar); label = L"K\,[s^2 \exp(-\!s)]")
	_ = lines!(ax, t_grid2, Kf_rec(x_funcs[2], s_grid2, t_grid2, A_mat_similar); label = L"K\,[s\, \exp(-\!s)]")
	# lines!(t_grid, f.(t_grid))
	axislegend(ax; position=:lt)
	# xlims!(ax, 0,1)
	ax.title = L("Forward problem solutions")
	fig
end

# ╔═╡ aebda2fa-a20d-4b59-8e88-569858e4d1bf
U2,S2,V2 = svd(A_mat_similar);

# ╔═╡ 411af349-7b26-48e4-b3cf-64a94d939635
minimum(S2)

# ╔═╡ bd739dcb-5640-4605-8bb1-37852a7a6ff0
with_backend(WGLMakie) do
	fig = Figure()
	ax = Axis(fig[1,1], yscale = log10)
	scatter!(ax, 1:length(S2), S2)
	# xlims!(ax, 0, length(S2))
	ax.title = L("Logplot of the singular values")
	fig
end

# ╔═╡ b5c837b5-c744-4d86-bf48-e7a11953d74a
md"### Relative error of the reconstruction"

# ╔═╡ be61de04-09b2-4744-a399-a5c5b0908dd8
with_backend(WGLMakie) do
	y_masked_1 = Kf_rec(x_funcs[1], s_grid2, t_grid2, A_mat_similar)
	y_masked_1_intp = scale(interpolate(y_masked_1, BSpline(Linear())), t_grid2)
	y_1 = Kf_rec(x_funcs[1], s_grid, t_grid)
	y_1_intp = scale(interpolate(y_1, BSpline(Linear())), t_grid)

	y_masked_2 = Kf_rec(x_funcs[2], s_grid2, t_grid2, A_mat_similar)
	y_masked_2_intp = scale(interpolate(y_masked_2, BSpline(Linear())), t_grid2)
	y_2 = Kf_rec(x_funcs[2], s_grid, t_grid)
	y_2_intp = scale(interpolate(y_2, BSpline(Linear())), t_grid)
	
	fig,ax,_ = lines(
		t_grid2, 
		abs.(y_1_intp.(t_grid2) .- y_masked_1_intp.(t_grid2))./abs.(y_1_intp.(t_grid2)); 
		label = L"\varepsilon\left[K\,[s^2 \exp(-\!s)]\right]"
	)
	_ = lines!(
		ax, t_grid2, 
		abs.(y_2_intp.(t_grid2) .- y_masked_2_intp.(t_grid2))./abs.(y_2_intp.(t_grid2)); 
		label = L"\varepsilon\left[K\,[s\, \exp(-\!s)]\right]"
	)
	
	axislegend(ax; position=:rt)
	# xlims!(ax, 0,1)
	ax.title = L("Relative error of the solution with the masked singularities")
	fig
end

# ╔═╡ 51522de7-5542-4607-b36a-8bf84b1fa538
md"""
# 3. Heat equation by discrete Fourier transform

We will use the variables $(s,t)$; $s$ is the spatial variable for initial condition $x(s)$, $t$ is the spatial variable for the terminal condition $y(t)$. Recall the formula

$$Kx(t) = y(t) = \sqrt{\frac{2}{\pi}} \sum_{n=1}^{\infty} c_n e^{-T n^2} \sin(nt),$$

with

$$x(s) = \sqrt{\frac{2}{\pi}} \sum_{n=1}^{\infty} c_n \sin(ns)$$

being the initial condition with Fourier (sine) coefficients $c_n$.

Put differently the operator $K$ is
```math
	y(t) = Kx(t) = \int_0^{\pi} k(t,s)x(s) \mathrm{d}s
```

with

$$k(t,s) = \frac{2}{\pi} \sum_{n=1}^{\infty} e^{-Tn^2} \sin(nt)\sin(ns).$$

``K`` is implemented by the discrete sine transform via FFT, e.g. in MATLAB by the following snippets

```matlab
% Fourier modes
k = [-N:N-1]; % Wave numbers
k = fftshift(k); %Shift frequencies to match FFT

% odd extension to force full Fourier series gives sine series
u = u0(2:end-1);
u_ext = [0,u,0,-flip(u)];

% Compute Fourier coeff
u_hat = fft(u_ext); % FFT of the solution
u_hatT = u_hat .* exp(-k.ˆ2 * T); % Multiply each Fourier coefficient by the right exponential

uend = ifft(u_hatT); % Inverse FFT to get back tophysical space
```
- Set up a complete code that given initial function $x$ approximates $y$ (both are sampled and represented as vectors)

- Experiment with different choices of initial conditions $x$ and compute an approximation of $y$. Look at different choices of $T$.

- The problem is linear and can be represented by a matrix $A$. Compute the matrix.

- Logplot the singular values of $A$.

- How would you characterize the method as a quadrature + collocation or expansion method.
"""

# ╔═╡ 7e890c9c-c613-40f5-b2f0-463e8cb156d0
md"---"

# ╔═╡ f41ac294-fdf4-4353-85e9-70a869bcfa46
x_0 = [
	s -> s^2*(1-s),
	s -> begin
		exp(-10 * (s - 0.8)^2) * sinpi(5s)^6 * s^2*(1-s) 
	end,
	s -> begin
		exp(-10 * (s - 0.5)^2) * cospi(5s)^6 * sinpi(s) / 7
	end
];

# ╔═╡ c6cfc2d5-77a2-453d-8be1-c9f5dea6e0e7
begin
	Base.string(_::typeof(x_0[1])) = "s^2(1 - s)"
	Base.string(_::typeof(x_0[2])) = raw"e^{-10(s - 0.8)^2} \sin(5\pi s)^6 s^2 (1-s)"
	Base.string(_::typeof(x_0[3])) = raw"e^{-10(s - 0.5)^2} \cos(5\pi s)^6 \sin(\pi s)/7"
end

# ╔═╡ 9548ecf0-41c1-4766-be03-a5df70682806
N3 = 1000;

# ╔═╡ e195f107-3284-44fc-a73d-77922a6b450b
ts = range(0,1, N3);

# ╔═╡ a021777e-edb8-43ff-9211-8cebb1cac4ab
function heat_forward(x,ts,T)
	u0 = x.(ts);

	N = length(ts)
	k_heat = -N:(N-1)
	k_heat = fftshift(k_heat)

	u = u0[2:end];
	u_ext = [0, u..., 0, (-1 .* reverse(u))...]
	u_hat = fft(u_ext);
	u_hatT = u_hat .* exp.(-k_heat.^2 * T)

	uend = ifft(u_hatT)[1:N]
	return real.(uend)
end;

# ╔═╡ ff8c8878-b2a8-4456-b22d-77ccf54cf12e
with_backend(WGLMakie) do
	fig = Figure()
	ax = Axis(fig[1,1])
	slg = SliderGrid(fig[2,1], (label = L"T", range = 0:0.01:0.5, startvalue = 0.03))
	T = slg.sliders[1].value
	for (x, st) in zip(x_0, [:solid,:dash,:dot])
		sol = @lift heat_forward(x,ts,$T)
		lines!(ax, ts, sol,
			#label = latexstring(raw"K\,[", string(x), "]"),
			color = :red, linestyle = st)
		lines!(ax, ts, x, label = latexstring(x), color = :blue, linestyle = st)
	end

	ax.title = L("Forward heat operator")
	
	axislegend(ax,position=:lt)
	# xlims!(ax, 0,1)
	fig
end

# ╔═╡ 2456f057-3d48-4844-b13c-5ce1e6e41c97
md"""
!!! todo
	Two last points
"""

# ╔═╡ 75cce3aa-0560-4311-b922-21d1365c396d
md"""
# 4. Deblurring problem by Galerkin method:

Define

```math
k(z) = \begin{cases}
1 + \cos(\pi z/3), & |z| < 3, \\
0, & |z| \geq 3.
\end{cases}
```

Approximate numerically the convolution operator $K$ in $L^2(-6, 6)$

```math
Kx(t) = \int_{-6}^6 k(t-s)x(s)ds \tag{1}
```

by a Galerkin scheme as follows:
- Take ``n`` with ``n/4`` integer and put ``h = 12/n``. Define the ``z``-grid by 
  ```math
  z_j = -6 + jh, \quad j = 0, \dots, n.
  ```

- Take for ``j = 1, \dots, n`` basis functions ``\phi_j(z) = \frac{1}{\sqrt{h}}`` for ``z \in (z_{j-1}, z_j)`` and 0 otherwise, and expand both ``x`` and ``y = Kx`` in that basis.

- Calculate the matrix elements
```math
  A_{ij} = \int_{-6}^{6} \int_{-6}^{6} \phi_i(t) k(t - s) \phi_j(s) \, ds \, dt.
```
  
!!! hint
	Start by fixing ``j - 1 = n/2`` leaving ``z_{j-1} = 0``.

- The matrix ``A`` now takes the expansion coefficients of ``x`` to expansion coefficients of ``y``. Interpret the matrix in the particular basis as a mapping from ``x`` sampled on the ``z``-grid to the corresponding ``y``.

"""

# ╔═╡ Cell order:
# ╠═5c07f4cc-71a5-11ef-3aec-efbcf53d4600
# ╟─a5c10b34-1424-4f63-bd1c-7fb35367cfba
# ╟─0ec62ab4-d32a-4488-b132-f4edcd642095
# ╟─621d8e84-9f44-4683-8c79-26558ad94efe
# ╟─fab1b14a-57bb-4cc1-8e61-dfa9448f235d
# ╠═388a8ac5-6932-4497-a482-7d8ebdc647f9
# ╠═99c02d8a-93f5-4ad1-94d3-40cfafaa25bc
# ╠═2fb5a6da-18dc-4053-b592-c6ae2f2d5a32
# ╟─51daabce-a894-4bd7-9be0-b23b29b067fb
# ╠═9cfedb8e-3d10-48b6-9544-e52bd4a67ef5
# ╟─b7c06aed-0df0-4b58-aa9b-6069f66700fc
# ╠═cde267d6-e522-414e-9db2-799dcd98782d
# ╠═a7986dc1-7b1e-40d5-a8ec-7d920f3c84eb
# ╠═5e5b4ea2-d4db-412a-9498-7dc1e16afef1
# ╠═de08d0cc-4291-4bd2-a23a-746e989b694e
# ╠═8940d452-c226-4035-8309-e719b6293d7e
# ╟─c3ecff37-af7b-48cd-9892-33c1864e6738
# ╟─40b721e1-9f10-44c3-85e1-e3174c7f083e
# ╠═ae6019a8-8a58-4ece-b373-822203b5664a
# ╟─51245408-d932-49b6-b444-466518e8e0ca
# ╠═f02ef0b6-28e2-413a-9168-038ed5dfbb18
# ╟─a72c1bd1-7210-428a-b190-90656db1d001
# ╠═8062f5b8-261d-41a1-959d-a5f34d154db1
# ╟─0dad41e6-0bfa-4987-a73f-672a42eb31a8
# ╠═8d77a0f8-ada4-4181-91c9-318103c741e1
# ╟─6c7c602b-8cf6-4f2e-b544-faa94f0146db
# ╠═aebda2fa-a20d-4b59-8e88-569858e4d1bf
# ╠═411af349-7b26-48e4-b3cf-64a94d939635
# ╟─bd739dcb-5640-4605-8bb1-37852a7a6ff0
# ╟─b5c837b5-c744-4d86-bf48-e7a11953d74a
# ╠═85fd322e-aaad-43d0-8c48-411108ffe108
# ╟─be61de04-09b2-4744-a399-a5c5b0908dd8
# ╟─51522de7-5542-4607-b36a-8bf84b1fa538
# ╟─7e890c9c-c613-40f5-b2f0-463e8cb156d0
# ╠═b5df0251-18ab-4e70-b087-4d245917e7ce
# ╠═f41ac294-fdf4-4353-85e9-70a869bcfa46
# ╠═c6cfc2d5-77a2-453d-8be1-c9f5dea6e0e7
# ╠═9548ecf0-41c1-4766-be03-a5df70682806
# ╠═e195f107-3284-44fc-a73d-77922a6b450b
# ╠═a021777e-edb8-43ff-9211-8cebb1cac4ab
# ╟─ff8c8878-b2a8-4456-b22d-77ccf54cf12e
# ╟─2456f057-3d48-4844-b13c-5ce1e6e41c97
# ╟─75cce3aa-0560-4311-b922-21d1365c396d
