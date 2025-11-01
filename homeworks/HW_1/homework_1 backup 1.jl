### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ a451693e-8249-11ef-0560-81bdef4ad34f
begin
    using MakieMaestro
    using MakieMaestro: MakieMaestro as MM
    using LinearAlgebra, LazyGrids, Optim
    using PlutoUI
    using BenchmarkTools
end;

# ╔═╡ 3048d64a-9a42-4ac3-850e-51a4f75f4913
begin
    using MakieMaestro.Units
    TEXT_WIDTH = Dict(:beamer => 307u"pt", :report => 512u"pt")
end

# ╔═╡ 2ddd358b-b6f5-467a-b642-3eda2c1c76a0
begin
    global plots = Dict()
    with_backend, with_backend! = MM.with_backend(plots)
end;

# ╔═╡ a8b835c8-9983-4edb-8b83-daca277c065b
MM.Themes.width!(TEXT_WIDTH[:report]);

# ╔═╡ 2bd815d4-ddad-462f-997b-081354643eba
MM.Themes.hwratio!(1);

# ╔═╡ 331f6552-39c4-4f68-9d6c-f4e280ceed03
MM.figure_dir!("src/figs/")

# ╔═╡ bfc8cf21-650b-49e7-84c8-f856560c3381
MM.Themes.update_theme!(:base, Makie.current_default_theme())

# ╔═╡ 4a5a9873-4405-47f6-a5e9-77fda77f61a9
md"""
# Homework 1

## 1.

Consider the backwards heat equation modelled by the forward operator
   
```math
	Kx(t) = \int_0^\pi x(s)k(t, s)ds \quad k(t, s) = \frac{2}{\pi} \sum_{n=1}^{\infty} e^{-T n^2} \sin(nt) \sin(ns).
```

- **(a)** We know that $K$ is compact in $L^2(0, \pi)$. Show that $K$ has the singular system given by
      
```math
	x_j(t) = \sqrt{\frac{2}{\pi}} \sin(jt), \quad y_j(s) = \sqrt{\frac{2}{\pi}} \sin(js), \quad \mu_j = e^{-T j^2}.
```

"""

# ╔═╡ 6d3baa8e-9ed9-4509-a4e9-cb73fc465fe7
md"""
- **(b)** Consider the implementation of $K$ by matrix $A \in \mathbb{R}^{M \times M}$ for $T = 0.05$. Compute the SVD of $A$. Compare in a plot the singular values of $K$ and $A$, and in a separate plot, the singular vectors for $A$ to the singular functions for $K$. Show by a numerical experiment that $Kx_j = \mu_j y_j$ for $j = 2, 4, 8$.
"""

# ╔═╡ 652772eb-c2ad-40d9-be02-c27f2632c9ae
M = 200

# ╔═╡ f4db248d-ea14-4447-9236-61800130ba04
T = 0.05

# ╔═╡ 6fa6519f-bbf0-4697-a11d-8b86af05a2dd
N = 16

# ╔═╡ 85438b17-0810-4968-bd60-aa5734a3bf52
ts, ss = range(π / M, π, M), range(π / M, π, M);

# ╔═╡ 0f1043ba-2c9e-4f29-ba7d-bcf10eaeef62
function A_alt(T, M, N)
    ts, ss, Ns = ndgrid(range(π / M, π, M), range(π / M, π, M), Base.OneTo(N))
    return dropdims(sum(exp.(-T * Ns .^ 2) .* sin.(ts .* Ns) .* sin.(ss .* Ns) *
                        2 / M
            ; dims=3), dims=3)
end

# ╔═╡ bfd667a0-1860-4573-914f-f8f053bd6a54
@benchmark A_alt(0.05, M, N)

# ╔═╡ a3ddae20-bfdf-47e1-bfaa-b2b7bb30d189
function A(T, M, N)
    ts, ss = ndgrid(range(π / M, π, M), range(π / M, π, M))
    A_mat = sum(zip(exp.(-T * Base.OneTo(N) .^ 2), map(n -> sin.(ts * n), Base.OneTo(N)), map(n -> sin.(ss * n), Base.OneTo(N)))) do (e, sint, sins)
        e * sint .* sins * 2 / M
    end
    return A_mat
end

# ╔═╡ 3b85f87f-00a9-4518-90d8-25614bb3d398
@benchmark A(0.05, M, N)

# ╔═╡ 4097533d-9a36-4da7-86d5-77da03676c08
@assert isapprox.(A(0.05, M, N), A_alt(0.05, M, N); atol=1e-10) |> all

# ╔═╡ 9f74acc7-1750-448f-8a68-4c384f172fc8
(A(0.05, M, N) .- A_alt(0.05, M, N)) |> extrema

# ╔═╡ 8227da2b-f083-4338-b42e-a4c0b717b220
norm((A(T, M, N) .- A(T, M, 100))) / norm(A(T, M, 100))

# ╔═╡ 09c2a765-0bb7-497b-8e4b-26b422ea5edf
A_0_05 = A(T, M, N);

# ╔═╡ 0a13927d-880c-4d96-b36b-0dbd5972edc8
U_A, Σ_A, V_A = svd(A_0_05);

# ╔═╡ 53a1c434-e0fb-4723-a467-93c91925e585
with_backend(WGLMakie) do
    f = Figure()
    ax = Axis(f[1, 1])
    for i in 1:2
        lines!(ts, U_A[:, i], label=latexstring(raw"U_{\cdot" * "$i}"))
        lines!(ts, V_A[:, i], label=latexstring(raw"V_{\cdot" * "$i}"))
    end
    axislegend(ax, position=:lt)
    f
end;

# ╔═╡ 8f375123-57cc-46d6-a28c-3ad109087a0a
μj(j) = exp.(-T * j^2)

# ╔═╡ 99418bd0-17b1-4b25-a252-a643640d4fec
xj(j) = t -> sqrt(2 / π)sin(j * t)

# ╔═╡ 8a0457a2-a606-4a38-a89b-eb50c029be39
yj(j) = s -> sqrt(2 / π)sin(j * s)

# ╔═╡ 91521b66-4278-4e7d-b395-60fe21ac7b90
function fig_singular_values(n=16)
    fig, ax, _ = scatterlines(1:n, μj.(1:n), label=L"\mu_j"; axis=(; yscale=log10), linestyle=:dash, alpha=0.5)
    ax.xlabel = L"j"
    ax.xticks = 1:n
    scatterlines!(ax, 1:n, Σ_A[1:n], label=L"\Sigma_{jj}", linestyle=:dot, marker=:utriangle, alpha=0.5)
    axislegend(ax, position=:rt)
    fig
end

# ╔═╡ e18fc8f9-6fae-45f6-9aa8-10faffa6e479
with_backend(fig_singular_values, WGLMakie)

# ╔═╡ a798cd57-ccad-4bf7-9644-ff630de9fef1
savefig(fig_singular_values, (16,), MM.Themes.get_width() / 2)

# ╔═╡ 997d1b77-7c2a-4af1-b9d1-e7cd4ef8bdce
md"""
!!! note
	The singular values are exactly the same...
"""

# ╔═╡ 82ad459a-12b6-41ee-811d-efca4e73d808
# due to rounding errors this becomes Inf at a certain point
[mean(xj(i).(ts) ./ V_A[:, i]) for i in 1:M]

# ╔═╡ efb9609c-fbb1-4ef6-b3b2-ba847a255223
sqrt(M / π) # normalization factor

# ╔═╡ 332597cc-f2a8-4273-8062-281597b9fcda
[norm(V_A[:, i]) for i in 1:M] # normalization

# ╔═╡ 59a8a840-5616-42a4-876a-d0949ac1b6b8
(V_A[:, 2] .* V_A[:, 1]) |> norm # orthogonality

# ╔═╡ 87410026-07bf-4d41-bae7-22c32a77772c
function fig_heatmap_singular_vectors(M=M, N=N, T=T)
    U_A, _, V_A = svd(A(T, M, N))
    ts = LinRange(π / M, π, M)
    X_ij = hcat([xj(i).(ts) for i in 1:M]...)

    max_abs = max(maximum(abs.(X_ij)), maximum(abs.(U_A)))
    colorrange = (-max_abs, max_abs)

    f, ax1, _ = heatmap(U_A; colormap=:tableau_red_blue, colorrange)
    ax1.title = L"U_A"
    ax1.aspect = DataAspect()

    ax2, hm = heatmap(f[1, 2], X_ij; colormap=:tableau_red_blue, colorrange)
    ax2.title = L"(x_j(t_i))_{ij}"
    ax2.aspect = DataAspect()
    ax1.xticks = ax2.xticks = (LinRange(0, M, 3) .+ [0.5, 0, 0], [L"0", L"\pi/2", L"\pi"])
    ax1.yticks = ax2.yticks = map(i -> i ≠ 1 ? i - 1 : 1, 1:10:M+1)
    ax2.yaxisposition = :right
    ax2.yticksvisible = ax2.yticklabelsvisible = false
    ax1.xlabel = ax2.xlabel = L"t"
    ax1.ylabel = ax = L"j"
    linkaxes!(ax1, ax2)
    Colorbar(f[1, 3], hm; tellheight=true)
    rowsize!(f.layout, 1, Aspect(2, 1))
    DataInspector(f)
    return f
end

# ╔═╡ ef98d416-2168-4ae7-ac80-148b8e48f2a8
with_backend(fig_heatmap_singular_vectors, WGLMakie, 100)

# ╔═╡ 3e6e560b-edac-428c-816c-6388dd6fce3a
savefig(() -> fig_heatmap_singular_vectors(100), "fig_heatmap_singular_vectors", MM.Themes.get_width() / 1.5, 0.6)

# ╔═╡ 27d0f237-4f24-4ac2-891a-3e965ae41823
function fig_compare_singular_vectors(n=3)
    f = Figure()
    ax = Axis(f[1, 1])
    ax.xlabel = L"t"
    ax.xticks = ([0, π / 2, π], [L"0", L"\pi/2", L"\pi"])
    for (clr, i) in zip(Makie.wong_colors(), 1:n)
        # scatter!(ax, ts, U_A[:,i], markersize = 2, color = clr, label = latexstring(raw"U_{\cdot" * "$i}"))
        scatter!(ax, ts, V_A[:, i], markersize=2, color=clr, label=latexstring(raw"U_{\cdot" * "$i}"))
        lines!(ax, 0 .. π, xj(i), color=clr, label=latexstring("x" * "_$i"))
    end
    xlims!(ax, 0 - 0.05, π + 0.05)
    axislegend(ax; position=:lb, nbanks=2)
    f
end

# ╔═╡ acfc2ae9-0302-4f27-81c7-63f3b23fa9e6
with_backend(fig_compare_singular_vectors, WGLMakie, 4)

# ╔═╡ 55dcc25f-2606-40b1-af24-3c1b9941dbdd
savefig(() -> fig_compare_singular_vectors(3), "fig_singular_vectors", MM.Themes.get_width() / 2)

# ╔═╡ e5f3c9ff-bbca-40ae-b27a-8db0de85db7d
# ╠═╡ disabled = true
#=╠═╡
(sing_vec_inds, vec_coefs) |> repr |> clipboard
  ╠═╡ =#

# ╔═╡ af3d32ab-31f4-4869-adaa-006b5f443b1c
begin
    sing_func(t; coefs=vec_coefs, inds=sing_vec_inds) =
        sum(zip(coefs, inds)) do (c, j)
            c * xj(j)(t)
        end
    exp_result(t; coefs=vec_coefs, inds=sing_vec_inds, T=T) =
        sum(zip(coefs, inds)) do (c, j)
            c * exp(-1 * T * j^2) * yj(j)(t)
        end
end

# ╔═╡ 1fe8e272-78c7-4707-83d7-f9fcd6d2e11c
function fig_sum_singular_demo()
    f, ax, _ = lines(0 .. π, sing_func, label=latexstring(raw"x = \sum_{k = 1}" * "^{$(maximum(sing_vec_inds))}a_k x_k(t)"))
    ax.xlabel = L"t"
    ax.xticks = ([0, π / 2, π], [L"0", L"\pi/2", L"\pi"])
    lines!(ax, 0 .. π, t -> exp_result.(t; T=0.01), label=L"y = Kx", color=:red)
    scatter!(ax, ts, A(0.01, M, N) * sing_func.(ss), markersize=4, color=:red, label=L"\hat{y}_i = A_{ij}\cdot x_j")
    axislegend(ax, position=:rb, nbanks=3)
    xlims!(ax, 0 - 0.05, π + 0.05)
    f
end

# ╔═╡ 86885244-d316-4abc-97c9-4f6dcaf64f80
with_backend(fig_sum_singular_demo, WGLMakie)

# ╔═╡ de0663d0-f355-4d74-a634-ba5666feff91
savefig(fig_sum_singular_demo, "fig_sum_singular_demo", MM.Themes.get_width() / 2)

# ╔═╡ f58de9d4-1ddc-4d92-805a-d9a273769e59
function fig_singular_demo(inds=[2, 4, 6])
    f = Figure()
    ax = Axis(f[1:3, 1])
    ax.xlabel = L"t"
    ax.xticks = ([0, π / 2, π], [L"0", L"\pi/2", L"\pi"])
    for (i, clr) in zip(inds, Makie.wong_colors())
        sing = t -> sing_func(t, coefs=[1], inds=[i])
        expect = t -> exp_result(t; coefs=[1], inds=[i], T=0.05)
        lines!(ax, 0 .. π, sing, label=latexstring("x" * "_$i(t)"), color=clr, linestyle=:dashdot)
        lines!(ax, 0 .. π, expect, label=latexstring(raw"\mu" * "_$i" * " y" * "_$i(t)"), color=clr)
        # scatter!(ax, ts, sing.(ss),  markersize = 4, color = :blue, label = L"x_t")
        scatter!(ax, ts, A(0.05, M, N) * sing.(ss), markersize=4, color=clr, label=latexstring(raw"A U_{\cdot" * "$i}"))
    end
    Legend(f[0, :], ax, nbanks=3, margin=(0, 0, 0, 0), tellwidth=false)
    xlims!(ax, 0 - 0.05, π + 0.05)
    rowgap!(f.layout, 1)
    f
end

# ╔═╡ f501d8cc-e599-4641-bc14-9af53f71a319
with_backend(fig_singular_demo, WGLMakie)

# ╔═╡ 772f160d-39cb-49c6-8678-5843fb19489c
savefig(fig_singular_demo, "fig_singular_demo", MM.Themes.get_width() / 2)

# ╔═╡ f31f4c05-9712-4a89-864b-1ea493dac85e
md"""

- **(c)** Fix the function
      
```math
	x(s) = s \sin(s) \exp((s - \pi/2)^2).
```

Plot (using the discrete approximations) the functions $x$ and $y = Kx$. Perturb $y$ by adding Gaussian noise $n$ of relative size $\|n\|_{L^2(0,\pi)}/\|y\|_{L^2(0,\pi)} = 2\%$ to obtain $y^\delta$, and make a Picard plot of the singular values $\mu_j$, the expansion coefficients $|(y^\delta, y_j)|$ and the reconstruction coefficients $|(y^\delta, y_j)|/\mu_j$.
"""

# ╔═╡ 9324fbfc-c28d-4269-9ec1-1641277c3d16
x(s) = s * sin(s)exp((s - π / 2)^2)

# ╔═╡ e977a8ee-b3ef-479c-9186-33bf93bbd26a
y_rec = A(T, M, N) * x.(ss);

# ╔═╡ 13329a29-55b7-40ca-baf7-1162839a9884
begin
    noise = randn(size(y_rec))
    noise .*= 0.02(norm(y_rec) / norm(noise))
end;

# ╔═╡ 33f283ad-d9bb-4545-a17b-891082bf3b36
function fig_fixed_function_noised()
    f, ax, _ = lines(0 .. π, x, label=L"x")
    scatterlines!(ss, y_rec; color=:red, label=L"Kx", marker=:x, markersize=2)
    lines!(ax, ss, y_rec .+ noise; label=L"y^\delta")
    # TODO: Instead of the scatter, can try errobars
    # errorbars!(ss, y_rec, noise; color = (:red, 0.5))
    # scatter!(ss, y_rec .+ noise; color = :red, markersize = 2)
    axislegend(ax; position=:lt)
    xlims!(ax, 0 - 0.05, π + 0.05)
    ax.xlabel = L"t"
    ax.xticks = ([0, π / 2, π], [L"0", L"\pi/2", L"\pi"])
    f
end

# ╔═╡ 3180d54e-acdc-4a26-87da-8499a274e815
with_backend(fig_fixed_function_noised, WGLMakie)

# ╔═╡ 13562ecd-fb57-43b3-94e3-b0b56d399cf6
savefig(fig_fixed_function_noised, "fig_fixed_function_noised", MM.Themes.get_width() / 2)

# ╔═╡ bbbfa69f-a633-4fd4-aa7c-6d49f981d976
R_power(n, y) = norm(n) / norm(y);

# ╔═╡ b8c0ee2f-ddd2-4f3a-a156-2ae477569a8f
R_power(noise, y_rec)

# ╔═╡ 43708ca7-4415-4595-a515-f344e9210b2b
y_coeff = U_A' * (y_rec .+ noise)

# ╔═╡ f83002a1-467f-4013-8491-645a7a14eda5
function fig_picard_plot(ran=1:40)
    f, ax, _ = scatter(ran, abs.(y_coeff[ran]), axis=(; yscale=log10), label=L"|(y^\delta, y_j\,)|")
    ax.xlabel = L"j"
    scatter!(ax, ran, abs.(y_coeff[ran]) ./ μj.(ran), label=L"{|(y^\delta, y_j)|}/{\mu_j}")
    scatter!(ax, ran, abs.((U_A'*y_rec)[ran]), label=L"|(y, y_j)|")
    axislegend(ax, position=:lt)
    ylims!(eps(), 10.0^35)
    xlims!(0.5, 40.5)
    ax.xticks = [1, 10, 20, 30, 40]
    f
end

# ╔═╡ da25ec1b-e4ba-466e-8165-be1c022eea0e
with_backend(fig_picard_plot, WGLMakie)

# ╔═╡ 163a4026-a723-4e6b-a884-d000b803aeda
savefig(fig_picard_plot, "fig_picard_plot", MM.Themes.get_width() / 2)

# ╔═╡ d1040246-21ef-4794-8491-cc7652fd70fb
md"""
## 2. 
Consider the Abel problem defined in $L^2(0, 1)$ by
```math
Kx(t) = \int_0^1 k(t, s)x(s)ds, \quad k(t, s) = \begin{cases} \frac{1}{\sqrt{t-s}}, & s < t \\ 0, & s > t. \end{cases}
```

   With your a numerical discretization based on a quadrature rule, set up the matrix $A$ corresponding to uniform $s$- and $t$-meshes with $n = 1000$ points. Fix the function

```math
	x(s) = s^2 - \cos(2\pi s).
```

   Discretize the function $x$ by a vector $x$, compute $y = Ax$ and add Gaussian noise of relative size $\delta = 10\%$ to obtain $y^\delta$

- **(a)** Compute TSVD approximation $x_k$ for well chosen values in $k \in [1, 100]$. Create a plot of reconstructions, and compare it to the ground truth $(x)$ via a plot of the relative error $\|x - x_k\|/\|x\|_2$.
"""

# ╔═╡ f0e408ac-9382-4aaa-9774-c2975e52e191
x2(s) = s^2 - cos(2π * s)

# ╔═╡ 83f0ae8e-68a9-41a1-9dad-6440b9789552
n = 1000

# ╔═╡ 28979f15-f119-4797-9d71-befe9b005813
begin
    ts2, ss2 = range(1 / n, 1, n), range(1 / n, 1, n)
    ss2 = ss2 .- step(ss2) / 2 # Shifting the grid to avoid the singularities
end;

# ╔═╡ 53e36ec0-745b-4086-bb42-335f06ab0639
function A2(n)
    ts, ss = range(1 / n, 1, n), range(1 / n, 1, n)
    ss = ss .- step(ss) / 2
    A_mat = [s < t ? 1 / sqrt(t - s) : 0 for t in ts, s in ss]
    return A_mat * step(ss)
end

# ╔═╡ 14f62c1b-5801-4d68-b467-f178c53db71e
y_rec2 = A2(n) * x2.(ts2);

# ╔═╡ aece29b1-d0cd-4cec-8c77-9c23c8ff6c9d
U_A2, Σ_A2, V_A2 = svd(A2(n));

# ╔═╡ 5c52e39a-eeb1-4eed-b99e-de9d482637cd
begin
    noise2 = randn(size(y_rec2))
    noise2 .*= 0.1(norm(y_rec2) / norm(noise2))
    y_delta2 = y_rec2 .+ noise2
end;

# ╔═╡ d03e467a-e71d-4653-aa4d-ef0911e7197f
function fig_abel_intro()
    f, ax, _ = lines(ts2, x2.(ts2), label=L"x")
    ax.xlabel = L"t"
    lines!(ax, ts2, y_delta2; label=L"y^\delta = y + n", linewidth=0.5)
    lines!(ax, ts2, y_rec2; label=L"y = Ax")
    xlims!(ax, 0 - 0.05 / π, 1 + 0.05 / π)
    axislegend(ax, position=:lt)
    f
end

# ╔═╡ 03640c0a-c231-4108-8172-907af5ece338
with_backend(fig_abel_intro, WGLMakie)

# ╔═╡ 9b005693-10e0-4fae-853b-c36831fd9086
savefig(fig_abel_intro, "fig_abel_intro", MM.Themes.get_width() / 2)

# ╔═╡ 4be55eae-a460-47e5-ad5c-49a6fa6bb909
y_coeff2 = U_A2' * y_delta2

# ╔═╡ 09946652-0e3b-4d7c-b2b5-f4660fa06a6d
y_coeff2_rec = y_coeff2 ./ Σ_A2

# ╔═╡ 59caf609-e338-4cd0-ae37-8fd8f7a43ec5
x_rec_TSVD(k) = V_A2 * vcat(y_coeff2_rec[1:k], zeros(n - k))

# ╔═╡ f4f72067-e42e-4d2f-a875-948bd0ae16d3
function fig_abel_TSVD(ks=20:20:100)
    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"t")
    for k in ks
        lines!(ts2, x_rec_TSVD(k); label=latexstring("x_" * "{$k}"))
    end
    axislegend(ax, position=:lt)
    xlims!(ax, 0 - 0.05 / π, 1 + 0.05 / π)
    f
end

# ╔═╡ 89a5dec6-8d2a-4ca6-8ee0-db8cce079d42
with_backend(fig_abel_TSVD, WGLMakie)

# ╔═╡ 6bbc3ee0-18cf-49a0-be92-ca48bbe4c8f3
savefig(fig_abel_TSVD, "fig_abel_TSVD", MM.Themes.get_width() / 2)

# ╔═╡ d867c623-0d1a-4358-8adb-5fb630c7903a
opt_k = argmin([norm(x2.(ts2) .- x_rec_TSVD(k)) / norm(x2.(ts2)) for k in 1:100])

# ╔═╡ 0f768ab7-fc7d-4231-9fbc-424ed923cb3e
function fig_abel_TSVD_relative_err()
    f, ax, _ = lines(1:100, k -> norm(x2.(ts2) .- x_rec_TSVD(k)) / norm(x2.(ts2)); label=L"\Vert x - x_k \,\Vert\, /\, \Vert x\, \Vert")
    ax.xlabel = L"k"
    scatter!(opt_k, norm(x2.(ts2) .- x_rec_TSVD(opt_k)) / norm(x2.(ts2)); color=:red, label=L"\min_k\{\Vert x - x_k \,\Vert\, /\, \Vert x\, \Vert\}")
    axislegend(ax; position=:rt)
    xlims!(ax, 0, 101)
    f
end

# ╔═╡ d2f935be-fff5-40b8-9bd4-ea866709e07d
with_backend(fig_abel_TSVD_relative_err, WGLMakie)

# ╔═╡ d3ad4536-26cb-4b71-b288-0bc2398295c4
savefig(fig_abel_TSVD_relative_err, "fig_abel_TSVD_relative_err", MM.Themes.get_width() / 2)

# ╔═╡ b446b1a6-b6d2-4e97-9979-38a43c6ddd3a
md"""

- **(b)** Compute Tikhonov solutions $x_\alpha$ for well chosen regularization parameters $\alpha \in (10^{-4}, 10)$. Plot solutions and compare to TSVD.
"""


# ╔═╡ 6fb7c50b-f951-4b0b-b352-056612d9378c
x_rec_Tikhonov(α) = V_A2 * (y_coeff2_rec .* Σ_A2 ./ (Σ_A2 .^ 2 .+ α))

# ╔═╡ d9bcd733-6076-4ee7-81a0-99e417e65ad0
σ2 = std(noise2)

# ╔═╡ 25193f07-9144-48d7-884d-8087dee46c16
function fig_abel_Tikhonov(αs=σ2 .* [3, 7, 10])
    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"t")
    for α in αs
        lines!(ts2, x_rec_Tikhonov(α); label=latexstring("x_" * "{$(round(α, sigdigits=3))}"), linewidth=0.5)
    end
    # lines!(ax, ts2, x2.(ts2), label = L"x")
    axislegend(ax, position=:lt)
    xlims!(ax, 0 - 0.05 / π, 1 + 0.05 / π)
    f
end

# ╔═╡ 143814d2-550f-467c-bd89-4fde933fed83
with_backend(fig_abel_Tikhonov, WGLMakie, σ2 .* [3, 7, 10])

# ╔═╡ 34d6cf67-93b6-46ab-88f9-3969cee77fb6
savefig(() -> fig_abel_Tikhonov(σ2 .* [3, 7, 10]), "fig_abel_Tikhonov", MM.Themes.get_width() / 2)

# ╔═╡ 8cf91f45-1a3a-466f-81a0-a1fc9320ec98
opt_α = optimize(α -> norm(x2.(ts2) .- x_rec_Tikhonov(α)) / norm(x2.(ts2)), [0.4], NelderMead())

# ╔═╡ b6a331d0-b15d-4240-8ba6-a496beb0d17f
function fig_abel_Tikhonov_relative_err()
    f, ax, _ = lines((3σ2) .. (10σ2), α -> norm(x2.(ts2) .- x_rec_Tikhonov(α)) / norm(x2.(ts2)); label=L"\Vert x - x_α \,\Vert\, /\, \Vert x\, \Vert")
    ax.xlabel = L"α"
    scatter!(Optim.minimizer(opt_α)[1], minimum(opt_α), label=L"\min_α\{\Vert x - x_α \,\Vert\, /\, \Vert x\, \Vert\}", color=:red)
    axislegend(ax; position=:ct)
    xlims!(ax, 0.2, 0.71)
    f
end

# ╔═╡ a2cdbee2-d6cd-4a8c-92dd-b84d1a0d5245
with_backend(fig_abel_Tikhonov_relative_err, WGLMakie)

# ╔═╡ 8348777c-23f8-4e33-86f5-d2fd06b6c718
savefig(fig_abel_Tikhonov_relative_err, "fig_abel_Tikhonov_relative_err", MM.Themes.get_width() / 2)

# ╔═╡ 0248a0b2-9978-4e77-b419-31b4f50fb50a
function fig_abel_compare()
    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"t")
    lines!(ax, ts2, x2.(ts2), label=L"x")
    α = Optim.minimizer(opt_α)[1]
    lines!(ts2, x_rec_Tikhonov(α); label=latexstring("x_" * "{$(round(α, sigdigits=3))}"), linewidth=0.5)
    lines!(ts2, x_rec_TSVD(opt_k); label=latexstring("x_" * "{$opt_k}"))
    axislegend(ax, position=:lt)
    xlims!(ax, 0 - 0.05 / π, 1 + 0.05 / π)
    f
end

# ╔═╡ 358ab404-e70e-4ef7-b7c6-084e6d6125a9
with_backend(fig_abel_compare, WGLMakie)

# ╔═╡ a926e135-a745-48fd-8779-77ecfd801744
savefig(fig_abel_compare, "fig_abel_compare", MM.Themes.get_width() / 2)

# ╔═╡ 2706b08e-ad09-4a4b-ad60-d10e527af014
md"""
- **(c)** Try the above with a discontinuous function $x$.
"""


# ╔═╡ 53ae9218-92a5-41fe-9265-ebad7cb3215f
x3(s) = (s > 0.5 ? -1 : 1) * (s^2 - cos(2π * s))

# ╔═╡ 4c7db5ba-8546-4ce0-8c67-c8cacf5308c1
y_rec3 = A2(n) * x3.(ts2);

# ╔═╡ 77c6a53e-03f7-4fb8-883e-1056a0323333
U_A3, Σ_A3, V_A3 = svd(A2(n));

# ╔═╡ e0c07aae-2ffe-436a-bb31-c88e004c8113
begin
    noise3 = randn(size(y_rec3))
    noise3 .*= 0.1(norm(y_rec3) / norm(noise3))
    y_delta3 = y_rec3 .+ noise3
end;

# ╔═╡ 036dd580-7a4e-4422-baa0-f27b0c3d2837
function fig_abel_discont_intro()
    f, ax, _ = lines(0 .. 0.5, x3)
    lines!((0.5 + eps()) .. 1, x3, color=Makie.wong_colors()[1], label=L"x")
    lines!(ax, ts2, y_delta3; label=L"y^\delta = y + n", linewidth=0.5)
    lines!(ax, ts2, y_rec3; label=L"y = Ax")

    axislegend(ax, position=:rt)
    xlims!(ax, 0 - 0.05 / π, 1 + 0.05 / π)
    ax.xlabel = L"t"
    f
end

# ╔═╡ 8b07738c-29ac-499b-a839-b87a2743637c
Makie.current_default_theme()

# ╔═╡ ec9e4a3f-0c09-4bad-b32d-4a45898c76fe


# ╔═╡ 1e652d2f-8218-406f-b507-14cafe0e77d0
with_backend(fig_abel_discont_intro, WGLMakie)

# ╔═╡ ab29ef52-586d-417a-8372-103bfd1cdc46
savefig(fig_abel_discont_intro, "fig_abel_discont_intro", MM.Themes.get_width() / 2)

# ╔═╡ 2e0a5266-81b9-4def-9e93-a2fdf0a8b095
y_coeff3 = U_A2' * y_delta3

# ╔═╡ 9fc0e3fd-a633-43fb-b50a-37886dcdd846
y_coeff3_rec = y_coeff3 ./ Σ_A3

# ╔═╡ f84e3b7c-8376-4175-8e43-c030ae1b1877
x_rec_TSVD_3(k) = V_A3 * vcat(y_coeff3_rec[1:k], zeros(n - k))

# ╔═╡ a10a3708-1ac7-47ef-84a3-f37ba42115bb
function fig_abel_discont_TSVD(ks=20:20:100)
    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"t")
    for k in ks
        lines!(ts2, x_rec_TSVD_3(k); label=latexstring("x_" * "{$k}"))
    end
    axislegend(ax, position=:lt)
    xlims!(ax, 0 - 0.05 / π, 1 + 0.05 / π)
    f
end

# ╔═╡ d704a312-efd8-4072-9fed-1b9b7c14b1cd
with_backend(fig_abel_discont_TSVD, WGLMakie)

# ╔═╡ 8d3982ea-c531-4f2e-a10b-72e6334d650a
savefig(fig_abel_discont_TSVD, "fig_abel_discont_TSVD", MM.Themes.get_width() / 2)

# ╔═╡ 32a7eaf8-499a-4402-a58c-a0a2390594fd
opt_k_3 = argmin([norm(x3.(ts2) .- x_rec_TSVD_3(k)) / norm(x3.(ts2)) for k in 1:100])

# ╔═╡ ad40ceb5-0da8-418c-8748-15a1370448f9
function fig_abel_discont_TSVD_relative_err()
    f, ax, _ = lines(1:100, k -> norm(x3.(ts2) .- x_rec_TSVD_3(k)) / norm(x3.(ts2)); label=L"\Vert x - x_k \,\Vert\, /\, \Vert x\, \Vert")
    ax.xlabel = L"k"
    scatter!(opt_k_3, norm(x3.(ts2) .- x_rec_TSVD_3(opt_k_3)) / norm(x3.(ts2)); color=:red, label=L"\min_k\{\Vert x - x_k \,\Vert\, /\, \Vert x\, \Vert\}")
    axislegend(ax; position=:rt)
    xlims!(ax, 0, 101)
    f
end

# ╔═╡ cefff7a1-629d-4d9d-9127-75693bc682a3
with_backend(fig_abel_discont_TSVD_relative_err, WGLMakie)

# ╔═╡ f87bd1c4-d57d-4c50-8857-eb4184f61f3d
savefig(fig_abel_TSVD_relative_err, "fig_abel_discont_TSVD_relative_err", MM.Themes.get_width() / 2)

# ╔═╡ 1a038e44-4e76-428d-a8cb-126c472a58b6
x_rec_Tikhonov_3(α) = V_A3 * (y_coeff3_rec .* Σ_A3 ./ (Σ_A3 .^ 2 .+ α))

# ╔═╡ f8fe7b61-ad2c-4700-b97f-1f577f5f131e
σ3 = std(noise3)

# ╔═╡ e769c1e8-44ab-4af3-8452-f314d7d3edc9
function fig_abel_discont_Tikhonov(αs=σ3 .* [3, 7, 10])
    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"t")
    for (α, clr) in zip(αs, Makie.wong_colors())
        lines!(ts2[ts2.<0.5], x_rec_Tikhonov_3(α)[ts2.<0.5]; label=latexstring("x_" * "{$(round(α, sigdigits=3))}"), linewidth=0.5, color=clr)
        lines!(ts2[ts2.>0.5], x_rec_Tikhonov_3(α)[ts2.>0.5]; linewidth=0.5, color=clr)
    end
    # lines!(ax, ts2, x2.(ts2), label = L"x")
    axislegend(ax, position=:lt)
    xlims!(ax, 0 - 0.05 / π, 1 + 0.05 / π)
    f
end

# ╔═╡ 89f98dfb-4362-416f-ae02-d450a3c6d06d
with_backend(fig_abel_discont_Tikhonov, WGLMakie, σ3 .* [3, 7, 10])

# ╔═╡ 384f8875-9531-4936-bee8-4d75ab993812
savefig(() -> fig_abel_discont_Tikhonov(σ3 .* [3, 7, 10]), "fig_abel_discont_Tikhonov", MM.Themes.get_width() / 2)

# ╔═╡ eff50726-f2d0-4f53-8a1b-2ee708ac9f2a
opt_α_3 = optimize(α -> norm(x3.(ts2) .- x_rec_Tikhonov_3(α)) / norm(x3.(ts2)), [0.4], NelderMead())

# ╔═╡ b61883bf-cd30-4ce4-a5c7-a0365511e241
function fig_abel_discont_Tikhonov_relative_err()
    f, ax, _ = lines((3σ3) .. (10σ3), α -> norm(x3.(ts2) .- x_rec_Tikhonov_3(α)) / norm(x3.(ts2)); label=L"\Vert x - x_α \,\Vert\, /\, \Vert x\, \Vert")
    ax.xlabel = L"α"
    scatter!(Optim.minimizer(opt_α_3)[1], minimum(opt_α_3), label=L"\min_α\{\Vert x - x_α \,\Vert\, /\, \Vert x\, \Vert\}", color=:red)
    axislegend(ax; position=:ct)
    xlims!(ax, 0.145, 0.505)
    f
end

# ╔═╡ 76233477-5552-4edb-8bac-c9804e425b14
with_backend(fig_abel_discont_Tikhonov_relative_err, WGLMakie)

# ╔═╡ e10a0096-4d93-473e-949c-09ea25a966b8
savefig(fig_abel_discont_Tikhonov_relative_err, "fig_abel_discont_Tikhonov_relative_err", MM.Themes.get_width() / 2)

# ╔═╡ a2e124e3-78c7-4f65-ba8e-411bb19a3ca4
function fig_abel_discont_compare()
    f = Figure()
    ax = Axis(f[1, 1], xlabel=L"t")
    lines!(0 .. 0.5, x3)
    lines!((0.5 + eps()) .. 1, x3, color=Makie.wong_colors()[1], label=L"x")

    α = Optim.minimizer(opt_α_3)[1]
    lines!(ts2[ts2.<0.5], x_rec_Tikhonov_3(α)[ts2.<0.5]; label=latexstring("x_" * "{$(round(α, sigdigits=3))}"), linewidth=0.5, color=Makie.wong_colors()[2])
    lines!(ts2[ts2.>0.5], x_rec_Tikhonov_3(α)[ts2.>0.5]; linewidth=0.5)
    lines!(ts2, x_rec_TSVD_3(opt_k_3); label=latexstring("x_" * "{$opt_k_3}"),
        color=Makie.wong_colors()[3])
    axislegend(ax, position=:lt)
    xlims!(ax, 0 - 0.05 / π, 1 + 0.05 / π)
    f
end

# ╔═╡ 4f2c1cee-c2d6-4513-8b0b-1d5941c1833c
with_backend(fig_abel_discont_compare, WGLMakie)

# ╔═╡ 5dff6d4b-c8fc-4f80-879f-20d84f9c3c13
savefig(fig_abel_discont_compare, "fig_abel_discont_compare", MM.Themes.get_width() / 2)

# ╔═╡ c3f747db-9419-4533-934b-047eaef98aa5
md"""
## 3. 
Let $K$ be a compact from Hilbert space $X$ to Hilbert space $Y$ with singular system ${x_j, y_j, \mu_j}$. Assume that $K$ is injective. Let $R_\alpha$ be the regularization operator for $K$ based on Tikhonov regularization written in the singular system for $K$ as

```math
x_\alpha = R_\alpha y = \sum_j \frac{\mu_j}{\mu_j^2 + \alpha} (y, y_j)x_j.
```

- **(a)** Show that $\lim_{\alpha \to 0} x_\alpha = x$.

!!! hint 
	Use the expansion above to consider $\|R_\alpha Kx - x\|^2$. Take the limit $\alpha \to 0$; be careful when interchanging the limit and summation.
"""

# ╔═╡ 81ddcfb5-2ea3-4260-aad2-b3b86687e5c9
md"""

- **(b)** Consider the deblurring model problem with solution $x$, matrix $A$, $y = Ax$, and $y^\delta = y + d$ with $\|d\| < \delta$. Choose the Tikhonov regularization parameter $\alpha(\delta)$ such that

```math
	\lim_{\delta \to 0} \alpha(\delta) = 0, \quad \lim_{\delta \to 0}\frac{\delta^2}{\alpha(\delta)} = 0.
```

Demonstrate numerically that $\lim_{\delta \to 0} \|R_{\alpha(\delta)}y^\delta - x\| = 0$.

!!! hint
	Take for instance $\delta = 0.01 : 0.01 : 0.1$ and $\alpha(\delta) = \delta$.
"""

# ╔═╡ e68ef930-e29c-4e54-8d7a-62510e3f257b
md"""
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
"""

# ╔═╡ a4b44d0c-0372-4e30-864f-0ff9e0cceff2
N4 = 100

# ╔═╡ dd3c4d74-d273-40ad-a60b-29ab9ded7f89
ts4 = ss4 = range(-6, 6, length=N4);

# ╔═╡ 81e9ea13-d979-4b66-b27f-72b8f09f4eb4
begin
    local ts, ss = ndgrid(ts4, ss4)
    local k(z) = abs(z) < 3 ? 1 + cospi(z / 3) : 0
    A4 = k.(ts .- ss) * step(ss4)
end;

# ╔═╡ bcffd625-10ff-4a94-9b6c-b5e694de8007
x4 = s -> begin
    s = (s + 6) / 12
    exp(-15 * (s - 0.6)^2) * sinpi(7s)^6 * s^2 * (3 - s)
end

# ╔═╡ 05790e7e-a29b-4723-a905-cc678efb8fec
function noise4(power, signal)
    n = randn(N4)
    n .*= power * norm(signal) / norm(n)
    return signal .+ n
end

# ╔═╡ 1a087812-fb7c-40e9-aea1-f0681662fb40
y_rec4 = A4 * x4.(ts4)

# ╔═╡ 5f549e03-03b9-456e-ab8c-c9ac00d2d70d
Base.string(_::typeof(x4)) = raw"e^{-15(s - 0.6)^2} \sin(7\pi s)^6 s^2 (3-s)"

# ╔═╡ 6fbf68c6-4f6a-4cb0-8dfb-f99304bc58cb
with_backend(WGLMakie) do
    f, ax, _ = lines(ts4, x4; label=L"x")
    lines!(ts4, y_rec4; label=L"y")
    axislegend(ax; position=:lt)
    xlims!(-6.2, 6.2)
    f
end

# ╔═╡ a999bf44-bc69-4f95-8106-573debef123a
R(α) = y_noisy -> begin
    A_ext = vcat(A4, I(N4) * sqrt(α))
    y_ext = vcat(y_noisy, zeros(N4))
    return A_ext \ y_ext
end

# ╔═╡ a0f8b656-d04b-4b9e-94fa-4c29ea9df569
function fig_limit_demo()
    f, ax, _ = lines(ts4, x4; label=L"x")
    lines!(ts4, y_rec4; label=L"y")
    for p in [0.1, 0.01, 0.001, 0.0001, 0.00001]
        x_alpha = R(p)(noise4(p, y_rec4))
        lines!(ts4, x_alpha, label=latexstring("x_" * "{$p}"), color=:red, linewidth=0.5, linestyle=:dot)
    end
    axislegend(ax; position=:lt)
    xlims!(-6.2, 6.2)
    f
end

# ╔═╡ c45d9320-2d26-4fe7-bcea-dfe8668030a4
with_backend(fig_limit_demo, WGLMakie)

# ╔═╡ 9366cac5-2c78-4e18-8883-da716d9afdc7
savefig(fig_limit_demo, "fig_limit_demo", MM.Themes.get_width() / 2)

# ╔═╡ ed5290a2-ea35-48a5-b213-44fd09b0937e
function fig_numerical_limit()
    x = x4.(ts4)
    f, ax, _ = lines(eps() .. 0.01, p -> norm(R(p)(noise4(p, y_rec4)) .- x) / norm(x); axis=(; xscale=sqrt, xreversed=true), label=L"\Vert x_{\alpha(\delta)} - x\Vert/\Vert x \Vert")
    ax.xticks = [0.01, 0.005, 0.0]
    ax.xlabel = L"\delta"
    axislegend(ax, position=:lb)
    xlims!(0.01, 0)
    f
end

# ╔═╡ 2e0cc0ec-4c06-42a3-81ec-225895a38bfa
with_backend(fig_numerical_limit, WGLMakie)

# ╔═╡ e2183ee0-bf95-4856-ba0c-a13be432cc70
savefig(fig_numerical_limit, "fig_numerical_limit", MM.Themes.get_width() / 2)

# ╔═╡ efd4c802-32bd-4a2a-bcdc-85150a43a1e5
md"""
### Important notes:
- Deadline for the HW1 is Friday October 11.
- All homeworks must be approved in order to qualify for the exam.
- Demonstrate in your solution your understanding of theory and computations.
- Each student should hand in a single pdf via Learn. The number of pages for the main report must not exceed 5. Code can be put in an appendix.
- Comment on all results: What do you see, and why is it reasonable (or unreasonable)?
- Make quality plots with proper names on axis, legends and captions. Zoom in on the relevant stuff.
"""

# ╔═╡ 81c9adb3-5408-40b8-9793-4af51d1ada12
sing_vec_inds, vec_coefs = ([1, 5, 6, 7, 14], [-1.1, -1.6, -1.1, -1.0, 0.8])

# ╔═╡ 3224dba9-a370-406f-ad81-a2a16ff52402
# ╠═╡ disabled = true
#=╠═╡
begin
	local n = 6
	sing_vec_inds = rand(1:N, n) |> sort |> unique
	vec_coefs = round.(randn(length(sing_vec_inds)); sigdigits = 2)
end;
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LazyGrids = "7031d0ef-c40d-4431-b2f8-61a8d2f650db"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MakieMaestro = "c922e6f1-a74a-406e-9215-09b2647b2648"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.6.0"
LazyGrids = "~1.0.0"
MakieMaestro = "~0.1.0"
Optim = "~1.9.4"
PlutoUI = "~0.7.60"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "eeeb14d3740faf42aa3ab3c3d3e98bd9bead4b7f"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "0ba8f4c1f06707985ffb4804fdad1bf97b233897"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.41"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.AccessorsExtra]]
deps = ["Accessors", "CompositionsBase", "ConstructionBase", "DataPipes", "InverseFunctions", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "34a9e3505307b3c318ee5478a9177f83f604ad16"
uuid = "33016aad-b69d-45be-9359-82a41f556fd4"
version = "0.1.92"

    [deps.AccessorsExtra.extensions]
    ColorTypesExt = "ColorTypes"
    DatesExt = "Dates"
    DictArraysExt = "DictArrays"
    DictionariesExt = "Dictionaries"
    DistributionsExt = "Distributions"
    DomainSetsExt = "DomainSets"
    FlexiMapsExt = "FlexiMaps"
    FlexiMapsStructArraysExt = ["FlexiMaps", "StructArrays"]
    SkipperExt = "Skipper"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TablesExt = "Tables"
    TestExt = "Test"
    URIsExt = "URIs"
    UnitfulExt = "Unitful"

    [deps.AccessorsExtra.weakdeps]
    ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    DictArrays = "e9958f2c-b184-4647-9c5a-224a61f6a14b"
    Dictionaries = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    DomainSets = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
    FlexiMaps = "6394faf6-06db-4fa8-b750-35ccc60383f7"
    Skipper = "fc65d762-6112-4b1c-b428-ad0792653d81"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "534820940e4359c09adc615f8bd06ca90d508ba6"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "4.0.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "0afa2b4ac444b9412130d68493941e1af462e26a"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.12.18"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "e771a63cc8b539eca78c85b0cabd9233d6c8f06f"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "26ec26c98ae1453c692efded2b17e15125a5bea1"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.28.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataManipulation]]
deps = ["Accessors", "AccessorsExtra", "DataPipes", "Dictionaries", "FlexiGroups", "FlexiMaps", "InverseFunctions", "Reexport", "Skipper", "StructArrays"]
git-tree-sha1 = "179eb5a4ae15f190e24116915d8141ab8572af76"
uuid = "38052440-ad76-4236-8414-61389b2c5143"
version = "0.1.19"
weakdeps = ["IntervalSets"]

    [deps.DataManipulation.extensions]
    IntervalSetsExt = "IntervalSets"

[[deps.DataPipes]]
git-tree-sha1 = "29077a8d5c093f4e0988e92c0d76f56c4c581900"
uuid = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
version = "0.3.18"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "5620ff4ee0084a6ab7097a27ba0c19290200b037"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.4"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "1cdab237b6e0d0960d5dcbd2c0ebfa15fa6573d9"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.4"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "03aa5d44647eaec98e1920635cdfed5d5560a8b9"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.117"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.Extents]]
git-tree-sha1 = "063512a13dbe9c40d999c439268539aa552d1ae6"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "8cc47f299902e13f90405ddb5bf87e5d474c0d38"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FlexiGroups]]
deps = ["AccessorsExtra", "Combinatorics", "DataPipes", "Dictionaries", "FlexiMaps"]
git-tree-sha1 = "d7321d77258809adc6f9c30811aef7c08d810cfb"
uuid = "1e56b746-2900-429a-8028-5ec1f00612ec"
version = "0.1.27"

    [deps.FlexiGroups.extensions]
    AxisKeysExt = "AxisKeys"
    CategoricalArraysExt = "CategoricalArrays"
    OffsetArraysExt = "OffsetArrays"
    StructArraysExt = "StructArrays"

    [deps.FlexiGroups.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[[deps.FlexiMaps]]
deps = ["Accessors", "DataPipes", "InverseFunctions"]
git-tree-sha1 = "bb989c35ee1f21ad9bda634540f0fafe89f199b2"
uuid = "6394faf6-06db-4fa8-b750-35ccc60383f7"
version = "0.1.27"

    [deps.FlexiMaps.extensions]
    AxisKeysExt = "AxisKeys"
    DictionariesExt = "Dictionaries"
    IntervalSetsExt = "IntervalSets"
    StructArraysExt = "StructArrays"
    UnitfulExt = "Unitful"

    [deps.FlexiMaps.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dictionaries = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "d52e255138ac21be31fa633200b65e4e71d26802"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW]]
deps = ["GLFW_jll"]
git-tree-sha1 = "7ed24cfc4cb29fb10c0e8cca871ddff54c32a4c3"
uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"
version = "3.4.3"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GLMakie]]
deps = ["ColorTypes", "Colors", "FileIO", "FixedPointNumbers", "FreeTypeAbstraction", "GLFW", "GeometryBasics", "LinearAlgebra", "Makie", "Markdown", "MeshIO", "ModernGL", "Observables", "PrecompileTools", "Printf", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "8753fba3356131357b5cd02500fe80c3668535d0"
uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
version = "0.10.18"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "8e233d5167e63d708d41f87597433f59a0f213fe"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.4"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "294e99f19869d0b0cb71aef92f19d03649d028d5"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.4.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "dc6bed05c15523624909b3953686c5f5ffa10adc"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "2bd56245074fab4015b9174f24ceba8293209053"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.27"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "LinearAlgebra", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "eb6ca9aef11db0c08b7ac0a5952c6c6ba6fbebf0"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.22"
weakdeps = ["DiffRules", "ForwardDiff", "IntervalSets", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.KwdefHelpers]]
deps = ["Accessors"]
git-tree-sha1 = "5078865e4949bb95d4dd2197504176ca849f7c48"
uuid = "80d9ef48-13f8-4f87-9333-d4c97b041895"
version = "0.1.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyGrids]]
deps = ["Statistics"]
git-tree-sha1 = "1f1bf025f03f644e6c18f458db813fe2e3a68b5d"
uuid = "7031d0ef-c40d-4431-b2f8-61a8d2f650db"
version = "1.0.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "be3051d08b78206fb5e688e8d70c9e84d0264117"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.21.18"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "9019b391d7d086e841cbeadc13511224bd029ab3"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.8.12"

[[deps.MakieExtra]]
deps = ["AccessorsExtra", "DataManipulation", "DataPipes", "InverseFunctions", "KwdefHelpers", "Makie", "ObjectiveC", "PyFormattedStrings", "Reexport", "StructHelpers"]
git-tree-sha1 = "3701fa21772ff68963cf101ef72a1cec76356110"
uuid = "54e234d5-9986-40d8-815f-a5e42de435f6"
version = "0.1.47"
weakdeps = ["GLMakie"]

    [deps.MakieExtra.extensions]
    GLMakieExt = "GLMakie"

[[deps.MakieMaestro]]
deps = ["CairoMakie", "ColorSchemes", "ColorTypes", "GLMakie", "InteractiveUtils", "LaTeXStrings", "Makie", "MakieExtra", "Markdown", "OffsetArrays", "Reexport", "Serialization", "Unitful", "WGLMakie"]
git-tree-sha1 = "18394b8528597421d1838abb3306a820c521c7c2"
uuid = "c922e6f1-a74a-406e-9215-09b2647b2648"
version = "0.1.0"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "f45c8916e8385976e1ccd055c9874560c257ab13"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.2"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "14a12d9153b1a1a22d669eede58b2ea2164ff138"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.4.13"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.ModernGL]]
deps = ["Libdl"]
git-tree-sha1 = "ac6cb1d8807a05cf1acc9680e09d2294f9d33956"
uuid = "66fc600b-dfda-50eb-8b99-91cfa97b1301"
version = "1.1.8"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.ObjectiveC]]
deps = ["CEnum", "Libdl", "Preferences"]
git-tree-sha1 = "f4d2579292a9b1866361b9ce746341be12ad0366"
uuid = "e86c9b32-1129-44ac-8ea0-90d5bb39ded9"
version = "3.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+3"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "966b85253e959ea89c53a9abebbf2e964fbf593b"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.32"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.PyFormattedStrings]]
deps = ["PrecompileTools", "Printf"]
git-tree-sha1 = "b38376d4f6a016461b905373eadf557147564f50"
uuid = "5f89f4a4-a228-4886-b223-c468a82ed5b9"
version = "0.1.12"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Skipper]]
git-tree-sha1 = "b71ca323b32136c7742f47704b648542d3367d4d"
uuid = "fc65d762-6112-4b1c-b428-ad0792653d81"
version = "0.1.14"

    [deps.Skipper.extensions]
    AccessorsExt = "Accessors"
    AxisKeysExt = "AxisKeys"
    DictionariesExt = "Dictionaries"
    MakieExt = "Makie"

    [deps.Skipper.weakdeps]
    Accessors = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dictionaries = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "02c8bd479d26dbeff8a7eb1d77edfc10dacabc01"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.11"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "9537ef82c42cdd8c5d443cbc359110cbb36bae10"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.21"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructHelpers]]
deps = ["ConstructionBase"]
git-tree-sha1 = "1d6b389c6d24e0fd76ba12c9c8111c9df197c1cd"
uuid = "4093c41a-2008-41fd-82b8-e3f9d02b504f"
version = "1.2.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "f21231b166166bebc73b99cea236071eb047525b"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.3"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "676bd14390033825be847e138108a1c53701407d"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.10.18"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56c6604ec8b2d82cc4cfe01aa03b00426aac7e1f"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.4+1"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "055a96774f383318750a1a5e10fd4151f04c29c5"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.46+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "d2408cac540942921e7bd77272c32e58c33d8a77"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.5.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc541bb19ed5b0ede95581fb2e41ecf179527d2"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.6.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╠═a451693e-8249-11ef-0560-81bdef4ad34f
# ╠═2ddd358b-b6f5-467a-b642-3eda2c1c76a0
# ╠═3048d64a-9a42-4ac3-850e-51a4f75f4913
# ╠═a8b835c8-9983-4edb-8b83-daca277c065b
# ╠═2bd815d4-ddad-462f-997b-081354643eba
# ╠═331f6552-39c4-4f68-9d6c-f4e280ceed03
# ╠═bfc8cf21-650b-49e7-84c8-f856560c3381
# ╟─4a5a9873-4405-47f6-a5e9-77fda77f61a9
# ╟─6d3baa8e-9ed9-4509-a4e9-cb73fc465fe7
# ╠═652772eb-c2ad-40d9-be02-c27f2632c9ae
# ╠═f4db248d-ea14-4447-9236-61800130ba04
# ╠═6fa6519f-bbf0-4697-a11d-8b86af05a2dd
# ╠═85438b17-0810-4968-bd60-aa5734a3bf52
# ╠═0f1043ba-2c9e-4f29-ba7d-bcf10eaeef62
# ╠═bfd667a0-1860-4573-914f-f8f053bd6a54
# ╠═a3ddae20-bfdf-47e1-bfaa-b2b7bb30d189
# ╠═3b85f87f-00a9-4518-90d8-25614bb3d398
# ╠═4097533d-9a36-4da7-86d5-77da03676c08
# ╠═9f74acc7-1750-448f-8a68-4c384f172fc8
# ╠═8227da2b-f083-4338-b42e-a4c0b717b220
# ╠═09c2a765-0bb7-497b-8e4b-26b422ea5edf
# ╠═0a13927d-880c-4d96-b36b-0dbd5972edc8
# ╠═53a1c434-e0fb-4723-a467-93c91925e585
# ╠═8f375123-57cc-46d6-a28c-3ad109087a0a
# ╠═99418bd0-17b1-4b25-a252-a643640d4fec
# ╠═8a0457a2-a606-4a38-a89b-eb50c029be39
# ╠═91521b66-4278-4e7d-b395-60fe21ac7b90
# ╠═e18fc8f9-6fae-45f6-9aa8-10faffa6e479
# ╠═a798cd57-ccad-4bf7-9644-ff630de9fef1
# ╟─997d1b77-7c2a-4af1-b9d1-e7cd4ef8bdce
# ╠═82ad459a-12b6-41ee-811d-efca4e73d808
# ╠═efb9609c-fbb1-4ef6-b3b2-ba847a255223
# ╠═332597cc-f2a8-4273-8062-281597b9fcda
# ╠═59a8a840-5616-42a4-876a-d0949ac1b6b8
# ╠═87410026-07bf-4d41-bae7-22c32a77772c
# ╠═ef98d416-2168-4ae7-ac80-148b8e48f2a8
# ╠═3e6e560b-edac-428c-816c-6388dd6fce3a
# ╠═27d0f237-4f24-4ac2-891a-3e965ae41823
# ╠═acfc2ae9-0302-4f27-81c7-63f3b23fa9e6
# ╠═55dcc25f-2606-40b1-af24-3c1b9941dbdd
# ╠═3224dba9-a370-406f-ad81-a2a16ff52402
# ╠═e5f3c9ff-bbca-40ae-b27a-8db0de85db7d
# ╠═81c9adb3-5408-40b8-9793-4af51d1ada12
# ╠═af3d32ab-31f4-4869-adaa-006b5f443b1c
# ╠═1fe8e272-78c7-4707-83d7-f9fcd6d2e11c
# ╠═86885244-d316-4abc-97c9-4f6dcaf64f80
# ╠═de0663d0-f355-4d74-a634-ba5666feff91
# ╠═f58de9d4-1ddc-4d92-805a-d9a273769e59
# ╠═f501d8cc-e599-4641-bc14-9af53f71a319
# ╠═772f160d-39cb-49c6-8678-5843fb19489c
# ╟─f31f4c05-9712-4a89-864b-1ea493dac85e
# ╠═9324fbfc-c28d-4269-9ec1-1641277c3d16
# ╠═e977a8ee-b3ef-479c-9186-33bf93bbd26a
# ╠═13329a29-55b7-40ca-baf7-1162839a9884
# ╠═33f283ad-d9bb-4545-a17b-891082bf3b36
# ╠═3180d54e-acdc-4a26-87da-8499a274e815
# ╠═13562ecd-fb57-43b3-94e3-b0b56d399cf6
# ╠═bbbfa69f-a633-4fd4-aa7c-6d49f981d976
# ╠═b8c0ee2f-ddd2-4f3a-a156-2ae477569a8f
# ╠═43708ca7-4415-4595-a515-f344e9210b2b
# ╠═f83002a1-467f-4013-8491-645a7a14eda5
# ╠═da25ec1b-e4ba-466e-8165-be1c022eea0e
# ╠═163a4026-a723-4e6b-a884-d000b803aeda
# ╟─d1040246-21ef-4794-8491-cc7652fd70fb
# ╠═f0e408ac-9382-4aaa-9774-c2975e52e191
# ╠═83f0ae8e-68a9-41a1-9dad-6440b9789552
# ╠═28979f15-f119-4797-9d71-befe9b005813
# ╠═53e36ec0-745b-4086-bb42-335f06ab0639
# ╠═14f62c1b-5801-4d68-b467-f178c53db71e
# ╠═aece29b1-d0cd-4cec-8c77-9c23c8ff6c9d
# ╠═5c52e39a-eeb1-4eed-b99e-de9d482637cd
# ╠═d03e467a-e71d-4653-aa4d-ef0911e7197f
# ╠═03640c0a-c231-4108-8172-907af5ece338
# ╠═9b005693-10e0-4fae-853b-c36831fd9086
# ╠═4be55eae-a460-47e5-ad5c-49a6fa6bb909
# ╠═09946652-0e3b-4d7c-b2b5-f4660fa06a6d
# ╠═59caf609-e338-4cd0-ae37-8fd8f7a43ec5
# ╠═f4f72067-e42e-4d2f-a875-948bd0ae16d3
# ╠═89a5dec6-8d2a-4ca6-8ee0-db8cce079d42
# ╠═6bbc3ee0-18cf-49a0-be92-ca48bbe4c8f3
# ╠═d867c623-0d1a-4358-8adb-5fb630c7903a
# ╠═0f768ab7-fc7d-4231-9fbc-424ed923cb3e
# ╠═d2f935be-fff5-40b8-9bd4-ea866709e07d
# ╠═d3ad4536-26cb-4b71-b288-0bc2398295c4
# ╟─b446b1a6-b6d2-4e97-9979-38a43c6ddd3a
# ╠═6fb7c50b-f951-4b0b-b352-056612d9378c
# ╠═d9bcd733-6076-4ee7-81a0-99e417e65ad0
# ╠═25193f07-9144-48d7-884d-8087dee46c16
# ╠═143814d2-550f-467c-bd89-4fde933fed83
# ╠═34d6cf67-93b6-46ab-88f9-3969cee77fb6
# ╠═8cf91f45-1a3a-466f-81a0-a1fc9320ec98
# ╠═b6a331d0-b15d-4240-8ba6-a496beb0d17f
# ╠═a2cdbee2-d6cd-4a8c-92dd-b84d1a0d5245
# ╠═8348777c-23f8-4e33-86f5-d2fd06b6c718
# ╠═0248a0b2-9978-4e77-b419-31b4f50fb50a
# ╠═358ab404-e70e-4ef7-b7c6-084e6d6125a9
# ╠═a926e135-a745-48fd-8779-77ecfd801744
# ╟─2706b08e-ad09-4a4b-ad60-d10e527af014
# ╠═53ae9218-92a5-41fe-9265-ebad7cb3215f
# ╠═4c7db5ba-8546-4ce0-8c67-c8cacf5308c1
# ╠═77c6a53e-03f7-4fb8-883e-1056a0323333
# ╠═e0c07aae-2ffe-436a-bb31-c88e004c8113
# ╠═036dd580-7a4e-4422-baa0-f27b0c3d2837
# ╠═8b07738c-29ac-499b-a839-b87a2743637c
# ╠═ec9e4a3f-0c09-4bad-b32d-4a45898c76fe
# ╠═1e652d2f-8218-406f-b507-14cafe0e77d0
# ╠═ab29ef52-586d-417a-8372-103bfd1cdc46
# ╠═2e0a5266-81b9-4def-9e93-a2fdf0a8b095
# ╠═9fc0e3fd-a633-43fb-b50a-37886dcdd846
# ╠═f84e3b7c-8376-4175-8e43-c030ae1b1877
# ╠═a10a3708-1ac7-47ef-84a3-f37ba42115bb
# ╠═d704a312-efd8-4072-9fed-1b9b7c14b1cd
# ╠═8d3982ea-c531-4f2e-a10b-72e6334d650a
# ╠═32a7eaf8-499a-4402-a58c-a0a2390594fd
# ╠═ad40ceb5-0da8-418c-8748-15a1370448f9
# ╠═cefff7a1-629d-4d9d-9127-75693bc682a3
# ╠═f87bd1c4-d57d-4c50-8857-eb4184f61f3d
# ╠═1a038e44-4e76-428d-a8cb-126c472a58b6
# ╠═f8fe7b61-ad2c-4700-b97f-1f577f5f131e
# ╠═e769c1e8-44ab-4af3-8452-f314d7d3edc9
# ╠═89f98dfb-4362-416f-ae02-d450a3c6d06d
# ╠═384f8875-9531-4936-bee8-4d75ab993812
# ╠═eff50726-f2d0-4f53-8a1b-2ee708ac9f2a
# ╠═b61883bf-cd30-4ce4-a5c7-a0365511e241
# ╠═76233477-5552-4edb-8bac-c9804e425b14
# ╠═e10a0096-4d93-473e-949c-09ea25a966b8
# ╠═a2e124e3-78c7-4f65-ba8e-411bb19a3ca4
# ╠═4f2c1cee-c2d6-4513-8b0b-1d5941c1833c
# ╠═5dff6d4b-c8fc-4f80-879f-20d84f9c3c13
# ╟─c3f747db-9419-4533-934b-047eaef98aa5
# ╟─81ddcfb5-2ea3-4260-aad2-b3b86687e5c9
# ╟─e68ef930-e29c-4e54-8d7a-62510e3f257b
# ╠═a4b44d0c-0372-4e30-864f-0ff9e0cceff2
# ╠═dd3c4d74-d273-40ad-a60b-29ab9ded7f89
# ╠═81e9ea13-d979-4b66-b27f-72b8f09f4eb4
# ╠═bcffd625-10ff-4a94-9b6c-b5e694de8007
# ╠═05790e7e-a29b-4723-a905-cc678efb8fec
# ╠═1a087812-fb7c-40e9-aea1-f0681662fb40
# ╠═5f549e03-03b9-456e-ab8c-c9ac00d2d70d
# ╠═6fbf68c6-4f6a-4cb0-8dfb-f99304bc58cb
# ╠═a999bf44-bc69-4f95-8106-573debef123a
# ╠═a0f8b656-d04b-4b9e-94fa-4c29ea9df569
# ╠═c45d9320-2d26-4fe7-bcea-dfe8668030a4
# ╠═9366cac5-2c78-4e18-8883-da716d9afdc7
# ╠═ed5290a2-ea35-48a5-b213-44fd09b0937e
# ╠═2e0cc0ec-4c06-42a3-81ec-225895a38bfa
# ╠═e2183ee0-bf95-4856-ba0c-a13be432cc70
# ╟─efd4c802-32bd-4a2a-bcdc-85150a43a1e5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
