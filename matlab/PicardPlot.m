% Example of Picard plot for Numerical differentiation problem
% Kim Knudsen, DTU Compute, 2024

%Call NumericalDifferentiation to get matrix A, yvec, yvecnoise and xvec
NumericalDifferentiation;
y = yvec;
yd= yvecnoise;


% Compute the SVD.
[U,S,V] = svd(A);
s = diag(S);

% Expansion coefficients of yd in U basis.
ycoeff = U'*yd;

%Picard plot
figure(3) , clf

semilogy(s)
hold on
semilogy(abs(ycoeff),'*','markersize',4);
semilogy(abs(ycoeff./s),'+','markersize',4)

legend('\mu_j','|u_i^T y_d|','|u_i^T y_d/\mu_j|')
title('Picard plot')
