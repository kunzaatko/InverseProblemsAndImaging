
% Produces a discretization, in the form of the square matrix A, of the
% blurring operator K in the Phillips test problem
%   Kf = y
% Method: expansion in terms of piecewise constant functions of value
% 1/sqrt(h) on non-overlapping intervals of length h ("top hat" functions).
%
% Matrix A for taking expansion coefficients of function f to
%         expansion coefficients of Kf = y.
%
% See also PhillipsScript.

% This code snip was written for the course 02624 at DTU.
% Per Christian Hansen & Kim Knudsen, DTU Compute, 2024


% number of points on interval (-6,6); n/4 must be positive integer.
n = 2^8;

% Setting up the matrix.
h = 12/n; n4 = n/4; r1 = zeros(1,n);
c = cos((-1:n4)*4*pi/n);
r1(1:n4) = h + 9/(h*pi^2)*(2*c(2:n4+1) - c(1:n4) - c(3:n4+2));
r1(n4+1) = h/2 + 9/(h*pi^2)*(cos(4*pi/n)-1);
A = toeplitz(r1);
