### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 83b6b8d6-3543-4cfa-98aa-243dceb55653
begin
    import Pkg
    Pkg.activate()
	using GLMakie, Makie, CairoMakie, WGLMakie
	using LinearAlgebra
	using PlutoUI
end;

# ╔═╡ 81fec091-bf07-438d-9b09-c3d256229562
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

# ╔═╡ Cell order:
# ╠═83b6b8d6-3543-4cfa-98aa-243dceb55653
# ╟─81fec091-bf07-438d-9b09-c3d256229562
