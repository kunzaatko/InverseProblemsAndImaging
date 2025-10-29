# We setup the linearized EIT problem for radial profile in unit ball

using MAT, Makie, LinearAlgebra, Optim


# Load precomputed eigenvalues for difference ND operator
# For instance using computeEigVal
b = matread("b.mat")["lambdaN"]

# Set up matrix
N = length(b)
Nvek = collect(1:N)

# Number of discrete points on r axis
M = 100
dr = 1 / M
rvec = dr:dr:1
rmat = Nvek * ones(1, M)
rmat .*= rvec'  # Equivalent to ones(N,1)*rvec in MATLAB

# This matrix approximates forward problem for linearized EIT
A = -2 * rmat .^ (2 .* Nvek .- 1) .* dr

# The linear discrete problem is now Ax=b
# Need to solve it via regularization

"""
Tikhonov (L2) Regularization
Solves the problem: min ||Ax - b||^2 + λ||x||^2
"""
function tikhonov_regularization(A, b, lambda)
    # Compute regularized solution using normal equations
    f(x) = (A * x .- b) .^ 2 |> sum
    opt = optimize(x -> f(x) + sqrt(lambda) * norm(x), randn(size(A, 2)))
    return opt.minimizer
end
