### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ da969514-affb-11ef-0c52-4bb9072ca4d3
begin
	import Pkg
	Pkg.activate()
	using LinearAlgebra, MAT, Optim
	using Optim: LineSearches
	using PlutoUI, ColorSchemes
	using MakieMaestro, LaTeXStrings
	using MakieMaestro: MakieMaestro as MM
	using MakieMaestro.Themes
end;

# ╔═╡ 17062bae-4423-4756-9a40-db7daca332e8
begin
	const TEXTWIDTH = 177u"mm"
	const FIGWIDTH = TEXTWIDTH * 0.8
	Themes.width!(FIGWIDTH)
	Themes.hwratio!(0.68)
	MM.figure_dir!("./src/figs")
end;

# ╔═╡ c4db05a7-d888-4ff0-8ef2-fbed85dfbf98
begin
	show_image = MM.Recipes.image
	show_image! = MM.Recipes.image!
end;

# ╔═╡ 18ccefbf-1bec-4c79-9d5f-a374428530b4
begin
	global plots = Dict()
	with_backend, with_backend! = MM.with_backend(plots)
end;

# ╔═╡ 77d3882c-7c5d-4823-ba0b-efd9c8d1bcc2
set_theme!(Themes.get_theme(:base))

# ╔═╡ c9862b40-6804-4176-a197-948a41f1f9e2
md"""
# Homework 3
"""

# ╔═╡ 76b06f26-51d0-4a5f-bca1-1266f2b45536
md"""
## Autoconvolution

Consider the autoconvolution operator ``K`` in ``L^2(0,1)`` defined for ``x \in L^2(0,1)`` by

```math
K(x)(t) = \int_0^t x(t-s)x(s)\,ds, \quad s \in (0,1)
```

with linearization around ``x_0 \in L^2(0,1)`` given by

```math
K'(x_0)h(t) = 2\int_0^t x_0(t-s)h(s)\,ds, \quad t \in (0,1).
```

+ **(a)** Take ``x(s) = 2e^{-2s}`` and show that ``y(t) = 4e^{-2t}``.
"""

# ╔═╡ 5207db32-ec6d-475e-89cf-f1294d752960
x(s) = 2exp(-2s)

# ╔═╡ 19c1075c-27a2-41d8-946a-61a75d0e24ba
y(t) = 4exp(-2t)*t

# ╔═╡ 189d4115-e783-4855-a4a6-4e9144ec203a
md"""
+ **(b)** Implement ``K`` numerically using a quadrature rule as a mapping from ``\mathbb{R}^{256}`` to ``\mathbb{R}^{256}``. Use your code to find an approximation of ``K(x)(t)`` and compare graphically to ``y(t)``.
"""

# ╔═╡ d7aabfa1-e37c-454f-9223-4d396da57879
function K(x, ts, ss)
	x_s = tril([x(t - s) for t in ts, s in ss]) .* step(ss)
	return map(eachrow(x_s)) do x_s_row 
		x_s_row' * x.(ss)
	end
end

# ╔═╡ 772f01dd-a742-4ba6-996a-ef83f213154e
function K(x::Vector)
	[sum(reverse(x[1:i]) .* x[1:i] .* 1/length(x)) for i in 1:length(x)]
end

# ╔═╡ 3d42124b-c3e4-4ce1-b32c-30def3b9fa09
ts = ss = range(0,1,256);

# ╔═╡ 04a93ffe-3515-4de1-9ac0-b1700dfeaa20
K(x; N=256) = K(x, range(0, 1, length = N), range(0, 1, length = N) .- 1/2N)

# ╔═╡ 5a1e1341-a487-49ae-8077-ace40139a9c1
K(x.(ts))

# ╔═╡ ab5b470c-b1a5-41ad-ae4d-696aaf426416
function fig_compare_quadrature_v_analytical_forward_autoconvolution()
	ss = range(0,1, length=256)
	f,ax,_ = lines(ss, K(x; N=256), label = L"K(x)(t)")
	ax.xlabel = L"t"
	lines!(ss, y, label = L"y(t)")
	axislegend(ax; position=:lt)
	f
end

# ╔═╡ 19f590cc-4e57-4d6d-9942-24f76931e9cd
with_backend(fig_compare_quadrature_v_analytical_forward_autoconvolution, WGLMakie)

# ╔═╡ 2e1fa772-4102-461e-97c1-64b0c123e35c
savefig(fig_compare_quadrature_v_analytical_forward_autoconvolution; update = true, width=FIGWIDTH/2)

# ╔═╡ e96df750-7b48-4bfd-934c-83c914f63e55
function fig_compare_quadrature_v_analytical_forward_autoconvolution_diff()
	ss = range(0,1, length=256)
	f,ax,_ = lines(ss, K(x; N=256) .- y.(ss), label = L"K(x)(t) - y(t)")
	ax.xlabel = L"t"
	axislegend(ax)
	f
end

# ╔═╡ e4a11e8e-12d5-44b1-9bd1-26ff3e8ab879
with_backend(fig_compare_quadrature_v_analytical_forward_autoconvolution_diff, WGLMakie)

# ╔═╡ b0418b67-3134-43b7-90cf-7474c67021c0
savefig(fig_compare_quadrature_v_analytical_forward_autoconvolution_diff; update=true, width=FIGWIDTH/2)

# ╔═╡ 9f257fa2-2ec2-4db1-bc83-e3bccf0f0d6e
md"""
+ **(c)** Add noise of relative size 5% to ``y`` to get ``y_\delta`` and consider solving ``K(x) = y_\delta`` by linearization: For the constant function ``x_0(t) = 2 - 1.5t``, discretize the operator ``K'(x_0)`` as a matrix ``A_0 \in R^{256\times256}``. Solve using Tikhonov regularization the linearized and discretized inverse problem to find an approximate solution ``x_1(t)``. Choose the regularization according to one of the parameter choice rules. Plot the solution and compare to ``x(t)``. Calculate the error.
"""

# ╔═╡ 2c149c74-d317-45fc-89c8-c26eb5550e50
begin
	δ = randn(256)
	y_δ = y.(ts) .+ 0.05 * norm(y.(ts))/norm(δ) .* δ
end

# ╔═╡ 1500c3b3-159a-4821-a9af-c12455afc6eb
with_backend(WGLMakie) do
	f,ax,_ = lines(ts, y_δ, label=L"y_\delta")
	ax.xlabel = L"t"
	axislegend(ax)
	f
end

# ╔═╡ 8c0d3671-971f-489f-84f6-db40a5a2eab3
function A_0(x_0, ts, ss)
	2 .* tril([x_0(t - s) for t in ts, s in ss]) .* step(ts)
end

# ╔═╡ 03c4c97d-076b-4f88-b45d-4ae35eebfddc
function A_0(x_i::Vector)
	2 .* [j-i > 0 ? x_i[j-i] : 0 for j in 1:length(x_i), i in 0:(length(x_i) - 1)] .* 1/length(x_i)
end

# ╔═╡ 8db2ce26-c52d-4bf3-965b-faaf9031f99a
A_0(x_0; N=256) = A_0(x_0, range(0,1,N), range(0,1,N))

# ╔═╡ 3bddc242-9839-4788-ab64-02ab7d745ca9
x_0(t) = 2 - 1.5t

# ╔═╡ a3ac057e-3bc4-4b42-9ca8-7a46ae5743b6
with_backend(WGLMakie) do
	f,ax,_ = lines(ts, A_0(x_0) * x.(ss); label=L"K'(x_0)[x](t)")
	lines!(ts, x_0, label=L"x_0(t)")
	ax.xlabel = L"t"
	axislegend(ax; position=:rc)
	f
end

# ╔═╡ d28d33a5-2095-4336-bd5f-83c601d4e588
function first_newton(α)
	y_0 = K(x_0)
	d_0 = y_δ .- y_0
	h_0 = [A_0(x_0);  √(α)*I(length(d_0))] \ vcat(d_0, zeros(length(d_0)))
	x_1 = x_0.(ts) .+ h_0
	y_0, d_0, h_0, x_1
end

# ╔═╡ c1aecbf6-f400-4ba6-bcd3-b253a820ef08
function fig_first_newton_step_slider()
	f = Figure() 
	
	slg = SliderGrid(f[2,1:6], (label=L"\alpha", range = 0.01:0.01:0.3, startvalue = 0.05))
	α = slg.sliders[1].value
	
	# y_0,d_0,h_0,x_1 
	l_res= lift(x -> first_newton(x), α)
	y_0 = lift(l -> l[1], l_res)
	d_0 = lift(l -> l[2], l_res)
	h_0 = lift(l -> l[3][:], l_res)
	x_1 = lift(l -> l[4], l_res)
	K_x_1 = lift(x -> K(x), x_1)
	
	ax,_ = lines(f[1,3:6], ts, y_0; label=L"y_0\coloneq K(x_0)")
	lines!(ts, y.(ts); label=L"y")
	lines!(ts, y_δ; label=L"y_\delta")
	lines!(ts, x_0.(ts); label=L"x_0")
	lines!(ts, x.(ts); label=L"x")
	lines!(ts, d_0; label=L"d_0 \coloneq y_\delta - K(x_0)")
	lines!(ts, h_0; label=L"h_0 \coloneq (K'[x_0])^{-1}(d_0)")
	lines!(ts, x_1; label=L"x_1 \coloneq x_0 + h_0")
	lines!(ts, K_x_1; label=L"y_1 \coloneq K(x_1)")
	ax.xlabel = L"t"
	Legend(f[1,1:2], ax)
	f
end

# ╔═╡ 5d17a635-8f9b-4d4a-8dd7-550417520fe5
with_backend(fig_first_newton_step_slider, CairoMakie);

# ╔═╡ f99df9f9-4d0a-4a65-be91-fdbe9a98574a
function fig_first_newton_step()
	f = Figure() 

	y_0,d_0,h_0,x_1 = first_newton(0.05)

	ax,_ = lines(f[1,3:6], ts, y_0; label=L"y_0\coloneq K(x_0)")
	lines!(ts, y.(ts); label=L"y")
	lines!(ts, y_δ; label=L"y_\delta")
	lines!(ts, x_0.(ts); label=L"x_0")
	lines!(ts, x.(ts); label=L"x")
	lines!(ts, d_0; label=L"d_0 \coloneq y_\delta - K(x_0)")
	lines!(ts, h_0; label=L"h_0 \coloneq (K'[x_0])^{-1}(d_0)")
	lines!(ts, x_1; label=L"x_1 \coloneq x_0 + h_0")
	lines!(ts, K(x_1); label=L"y_1 \coloneq K(x_1)")
	ax.xlabel = L"t"
	Legend(f[1,1:2], ax)
	f
end

# ╔═╡ c03a20c1-42c7-43f2-ac3b-9237a6ce1b29
with_backend(fig_first_newton_step, WGLMakie)

# ╔═╡ 1a909f2a-94af-4b53-9832-43d7d91cb888
savefig(fig_first_newton_step;width=FIGWIDTH/2, update=true)

# ╔═╡ 923ba046-698b-487f-a973-8ba99be81d4f
md"""
+ **(d)** Take another Newton step to find ``x_2(t)``.
"""

# ╔═╡ cad981d7-e6d8-448c-bb96-98b23dbdd5e7
function A_0_tikhonov(args...;α=0.05, vargs...)
	A_0_noreg = A_0(args...; vargs...)
	@assert size(A_0_noreg,1) == size(A_0_noreg, 2)
	A_0_reg = [A_0_noreg; √(α)*I(size(A_0_noreg,1))]
	return A_0_reg
end

# ╔═╡ 5221d392-0419-427d-a195-1935c6e5029f
function newton_step_tikhonov(x_i; α=0.05)
	y_i = K(x_i)
	d_i = y_δ .- y_i
	h_i =  A_0_tikhonov(x_i;α=α) \ vcat(d_i,zeros(length(d_i)))
	x_i .+ h_i
end

# ╔═╡ b829b70e-6013-4e47-a483-a6713f449159
begin
	x_is_reg = [x_0.(ts)]
	for _ in 1:4
		push!(x_is_reg, newton_step_tikhonov(last(x_is_reg); α=0.05))
	end
end

# ╔═╡ 52bdffc1-f063-4d31-862f-fd7c4aa5fdb9
function fig_newton_iterations_reg_lines()
	f = Figure()
	ax = Axis(f[1,1])
	for (i,x_i) in enumerate(x_is_reg[1:(end-1)])
		lines!(ts, x_i; color=ColorSchemes.Blues[(i+2)/(length(x_is_reg) + 2)], linestyle=:dash)
	end
	lines!(ts, last(x_is_reg); color=ColorSchemes.Blues[1.0], label=L"(\hat{x}_\alpha)_i", linestyle=:dash)
	lines!(ts, x.(ts); label=L"x", color = :orange)
	ax.xlabel = L"t"
	axislegend(ax)
	f
end

# ╔═╡ 907bd297-fca4-43e9-8fa1-7de4062224d2
with_backend(fig_newton_iterations_reg_lines, WGLMakie)

# ╔═╡ 8052a335-b29f-465a-b0eb-4c90b4b26b61
savefig(fig_newton_iterations_reg_lines; width=FIGWIDTH/2, update=true)

# ╔═╡ 67933e23-c3c7-4a8d-ae36-af6c96955472
function fig_newton_iterations_surf()
	f = Figure()
	ax = Axis3(f[1,1])
	x_surf = hcat(x_is_reg...)
	surface!(x_surf; color=hcat([fill(ColorSchemes.Blues[i/length(x_is_reg)], length(x_is_reg[1])) for i in 1:length(x_is_reg)]...), alpha=0.9, label=L"x_i")
	lines!(1:length(x_is_reg[1]), fill(length(x_is_reg), length(x_is_reg[1])), x.(ts); color=:orange, label=L"x")
	ax.xlabel = L"t"
	ax.xticklabelsvisible = false
	ax.yticklabelsvisible = false
	axislegend(ax)
	f
end

# ╔═╡ ae81d427-92d3-4c64-a074-66740249aad3
with_backend(fig_newton_iterations_surf, WGLMakie)

# ╔═╡ f737e182-cbe4-416a-b9a5-cf99dbbd1f48
md"""
+ **(e)** What happens if you start from the constant function ``x_0(t) = -2 + 1.5t`` instead?
"""

# ╔═╡ 4235dbe6-7b7d-4382-9c9d-dbfb8b1d69db
begin
	x_0_alt(t) = -2 + 1.5t
	x_is_reg_alt = [x_0_alt.(ts)]
	for _ in 1:15
		push!(x_is_reg_alt, newton_step_tikhonov(last(x_is_reg_alt); α=0.05))
	end
end

# ╔═╡ 9b901360-7ed5-452b-b874-eca2142cee28
function fig_newton_iterations_reg_lines_alt()
	f = Figure()
	ax = Axis(f[1,1])
	for (i,x_i) in enumerate(x_is_reg_alt[1:(end-1)])
		lines!(ts, x_i; color=ColorSchemes.Blues[(i+25)/(length(x_is_reg_alt) + 25)], linestyle=:dash)
	end
	lines!(ts, last(x_is_reg_alt); color=ColorSchemes.Blues[1.0], label=L"(\hat{x}_\alpha)_i", linestyle=:dash)
	lines!(ts, x.(ts); label=L"x", color = :orange)
	ax.xlabel = L"t"
	axislegend(ax)
	f
end

# ╔═╡ 42a02aa6-42d6-48b9-8d67-4ab6cf5ba29f
with_backend(fig_newton_iterations_reg_lines_alt, WGLMakie)

# ╔═╡ 4423faba-0cf3-47b6-b5e2-d107451af9fb
savefig(fig_newton_iterations_reg_lines_alt; width=FIGWIDTH/2, update=true)

# ╔═╡ 99469964-d3ec-433e-bc1c-ca68e322059c
function fig_newton_iterations_reg_lines_minus()
	f = Figure()
	ax = Axis(f[1,1])
	for (i,x_i) in enumerate(x_is_reg_alt[1:3])
		lines!(ts, x_i; color=ColorSchemes.Blues[(i+2)/(4 + 2)], linestyle=:dash)
	end
	lines!(ts, x_is_reg_alt[4]; color=ColorSchemes.Blues[1.0], label=L"(\hat{x}_\alpha)_i", linestyle=:dash)
	lines!(ts, -1 .* x.(ts); label=L"-x", color = :orange)
	ax.xlabel = L"t"
	axislegend(ax; position=:lt)
	f
end

# ╔═╡ 8f7af5fb-2a5e-452c-a3b1-2899f4459514
with_backend(fig_newton_iterations_reg_lines_minus, WGLMakie)

# ╔═╡ 61f28b62-fd02-40a9-9d55-6e936fb032e7
savefig(fig_newton_iterations_reg_lines_minus;width=FIGWIDTH/2, update=true)

# ╔═╡ 3fea10a2-ac6e-4d8b-bbec-acc84812437e
md"""
+ **(f)** Is the inverse problem ``(K(x) = y)`` ill-posed; is it locally ill-posed?
"""

# ╔═╡ 4a6d6def-d04b-4fe5-ba09-6d8c0db3e43f
md"""
## Radial Calderón problem

The radial FIT/Calderón problem linearized around ``\gamma_0 = 1`` is defined through the operator ``K`` as follows

```math
Kh(n) = -2\int_0^1 h(r)r^{2n-1}\,dr = \lambda_n, \tag{1}
```

with ``\lambda_n`` denoting eigenvalues of ``\Lambda_\gamma - \Lambda_1``.

+ **(a)** In the file `DHW3.mat` on inside you find 10 precomputed ('measured') eigenvalues for a particular ``\Lambda_\gamma - \Lambda_1``. Compute an estimate of ``\gamma = 1 + h`` using both Tikhonov and TV regularization and plot it.
"""

# ╔═╡ 0f8f3371-a2c6-4d8c-b97e-8da1521f21e1
λ_n_diff = matread("./bHW3.mat")["lambdaNdiff"][:]

# ╔═╡ 0cc2c5bd-e117-4019-af86-e44fda56c39d
function K_FIT(n=10; N=100)
	rs = range(0,1,N)
	-2 .* hcat([rs.^(2en - 1)' for en in 1:n]...) * step(rs) 
end

# ╔═╡ 6ae86bbe-4d7a-4f65-b471-37791e562bf9
x_tikhonov_reg(α; N=100) = [K_FIT(;N) * K_FIT(;N)'; √(α)I(N)] \ [(K_FIT(;N) * λ_n_diff); zeros(N)]

# ╔═╡ bb5d9821-55fa-4ecd-bf8f-e1441392cd40
K_FIT()' * randn(100)

# ╔═╡ 39f0ded3-c29c-4878-b3c7-8a081ded8def
function tikhonov_regularization(A, λ_n, α)
    # Compute regularized solution using normal equations
    f(h) = (A' * h .- λ_n).^ 2 |> sum
	x_0 = randn(size(A, 1))
    opt = optimize(h -> f(h) + √(α) * norm(h), randn(size(A,1)), Optim.Options(iterations=1_000_000))
    return opt
end

# ╔═╡ 05454e0f-f21a-4837-a3c1-64724fc24b4e
tikhonov_regularization(K_FIT(;N=50), λ_n_diff, 0.000005).minimizer

# ╔═╡ 0b80f02b-4dd7-4f3d-898a-7048749562e8
x_tikhonov_reg(0.5)

# ╔═╡ 0216bd6a-78eb-4f04-9cc9-3cb5585ec9e5
tikhonov_FIT(α=0.05,n=10) = [K_FIT(; N=512); √(α)*I(n)]' \ λ_n_diff

# ╔═╡ cbfc0d0e-4f92-443c-b6f4-e7eab33ddd1d
x_noreg(;N=50) = K_FIT(;N)' \ λ_n_diff

# ╔═╡ eaa15cef-513b-434d-8afe-c26600f90141
with_backend(WGLMakie) do 
	
	f = Figure()
	slg = SliderGrid(f[2,1], (label=L"\alpha", range=0:10^(-10):(5*10^(-10)), startvalue=0.25))
	α = slg.sliders[1].value
	sol = lift(x -> tikhonov_regularization(K_FIT(;N=50), λ_n_diff, x).minimizer, α)
	ran = lift(x -> range(0,1, length(x)), sol)
	ax,_ = lines(f[1,1], ran, sol, label="tikhonov")
	lines!(ran, x_noreg(), label="no regularization")
	axislegend(ax, position=:lt)
	# lines!(range(0,1,10), λ_n_diff)
	f
end

# ╔═╡ 83c85be6-9b28-44e0-8aa8-d9d7c0f8fb58
md"""
+  **(b)** Knowing that ``\gamma`` has the radial form

```math
\gamma(x) = \begin{cases}
    1 + \hat{q}, & |x| \leq R, \\
    1, & R < |x| \leq 1.
\end{cases}
```

give an estimate from the above of the contrast ``\hat{q}`` and radius ``R``.
"""

# ╔═╡ ad032ee3-0645-480a-b16d-8e83799450d3
begin
	R = 1
	M = 1000
	N = length(λ_n_diff)
	dr = 1/M
	rs = range(dr, R, M)
	r_mat = ones(N,1)*rs'
	N_vec = (1:N)'
	N_mat = N_vec' * ones(1, M)
	A = -2*r_mat.^(2*N_mat.-1).*dr
end

# ╔═╡ b4a79673-b1a2-4bf4-8dc1-529b328fa632
with_backend(WGLMakie) do 
	lines(range(0,1, length(A \ λ_n_diff)), A \ λ_n_diff)
end

# ╔═╡ Cell order:
# ╠═da969514-affb-11ef-0c52-4bb9072ca4d3
# ╠═17062bae-4423-4756-9a40-db7daca332e8
# ╠═c4db05a7-d888-4ff0-8ef2-fbed85dfbf98
# ╠═18ccefbf-1bec-4c79-9d5f-a374428530b4
# ╠═77d3882c-7c5d-4823-ba0b-efd9c8d1bcc2
# ╟─c9862b40-6804-4176-a197-948a41f1f9e2
# ╟─76b06f26-51d0-4a5f-bca1-1266f2b45536
# ╠═5207db32-ec6d-475e-89cf-f1294d752960
# ╠═19c1075c-27a2-41d8-946a-61a75d0e24ba
# ╟─189d4115-e783-4855-a4a6-4e9144ec203a
# ╠═d7aabfa1-e37c-454f-9223-4d396da57879
# ╠═772f01dd-a742-4ba6-996a-ef83f213154e
# ╠═3d42124b-c3e4-4ce1-b32c-30def3b9fa09
# ╠═5a1e1341-a487-49ae-8077-ace40139a9c1
# ╠═04a93ffe-3515-4de1-9ac0-b1700dfeaa20
# ╠═ab5b470c-b1a5-41ad-ae4d-696aaf426416
# ╠═19f590cc-4e57-4d6d-9942-24f76931e9cd
# ╠═2e1fa772-4102-461e-97c1-64b0c123e35c
# ╠═e96df750-7b48-4bfd-934c-83c914f63e55
# ╠═e4a11e8e-12d5-44b1-9bd1-26ff3e8ab879
# ╠═b0418b67-3134-43b7-90cf-7474c67021c0
# ╟─9f257fa2-2ec2-4db1-bc83-e3bccf0f0d6e
# ╠═2c149c74-d317-45fc-89c8-c26eb5550e50
# ╠═1500c3b3-159a-4821-a9af-c12455afc6eb
# ╠═8c0d3671-971f-489f-84f6-db40a5a2eab3
# ╠═03c4c97d-076b-4f88-b45d-4ae35eebfddc
# ╠═8db2ce26-c52d-4bf3-965b-faaf9031f99a
# ╠═3bddc242-9839-4788-ab64-02ab7d745ca9
# ╠═a3ac057e-3bc4-4b42-9ca8-7a46ae5743b6
# ╠═d28d33a5-2095-4336-bd5f-83c601d4e588
# ╠═c1aecbf6-f400-4ba6-bcd3-b253a820ef08
# ╠═5d17a635-8f9b-4d4a-8dd7-550417520fe5
# ╠═f99df9f9-4d0a-4a65-be91-fdbe9a98574a
# ╠═c03a20c1-42c7-43f2-ac3b-9237a6ce1b29
# ╠═1a909f2a-94af-4b53-9832-43d7d91cb888
# ╟─923ba046-698b-487f-a973-8ba99be81d4f
# ╠═cad981d7-e6d8-448c-bb96-98b23dbdd5e7
# ╠═5221d392-0419-427d-a195-1935c6e5029f
# ╠═b829b70e-6013-4e47-a483-a6713f449159
# ╠═52bdffc1-f063-4d31-862f-fd7c4aa5fdb9
# ╠═907bd297-fca4-43e9-8fa1-7de4062224d2
# ╠═8052a335-b29f-465a-b0eb-4c90b4b26b61
# ╠═67933e23-c3c7-4a8d-ae36-af6c96955472
# ╠═ae81d427-92d3-4c64-a074-66740249aad3
# ╟─f737e182-cbe4-416a-b9a5-cf99dbbd1f48
# ╠═4235dbe6-7b7d-4382-9c9d-dbfb8b1d69db
# ╠═9b901360-7ed5-452b-b874-eca2142cee28
# ╠═42a02aa6-42d6-48b9-8d67-4ab6cf5ba29f
# ╠═4423faba-0cf3-47b6-b5e2-d107451af9fb
# ╠═99469964-d3ec-433e-bc1c-ca68e322059c
# ╠═8f7af5fb-2a5e-452c-a3b1-2899f4459514
# ╠═61f28b62-fd02-40a9-9d55-6e936fb032e7
# ╟─3fea10a2-ac6e-4d8b-bbec-acc84812437e
# ╟─4a6d6def-d04b-4fe5-ba09-6d8c0db3e43f
# ╠═0f8f3371-a2c6-4d8c-b97e-8da1521f21e1
# ╠═0cc2c5bd-e117-4019-af86-e44fda56c39d
# ╠═6ae86bbe-4d7a-4f65-b471-37791e562bf9
# ╠═bb5d9821-55fa-4ecd-bf8f-e1441392cd40
# ╠═39f0ded3-c29c-4878-b3c7-8a081ded8def
# ╠═05454e0f-f21a-4837-a3c1-64724fc24b4e
# ╠═0b80f02b-4dd7-4f3d-898a-7048749562e8
# ╠═0216bd6a-78eb-4f04-9cc9-3cb5585ec9e5
# ╠═cbfc0d0e-4f92-443c-b6f4-e7eab33ddd1d
# ╠═eaa15cef-513b-434d-8afe-c26600f90141
# ╟─83c85be6-9b28-44e0-8aa8-d9d7c0f8fb58
# ╠═ad032ee3-0645-480a-b16d-8e83799450d3
# ╠═b4a79673-b1a2-4bf4-8dc1-529b328fa632
