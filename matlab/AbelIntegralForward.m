%Abel integral operator
% Course 02624
% Problem: Approximation of the Abel integral operator Kx(t) =  int_0^t x(s)/sqrt(t-s)ds as a matrix A using numerical quadrature on 
% off set s and t grids to avoid the singularity.
% Kim Knudsen 2024

  
%clear;
%close all;
 N= 128;
 h= 1/N;
 
 tvec = h/2:h:(1-h/2);
 svec = (tvec(1:end)+h/2)';
 
 tMatrix = repmat(tvec,N,1);
 sMatrix = repmat(svec,1,N);
 
 %Evaluate matrix A
 A = zeros(N,N);
 index = tMatrix<sMatrix;
 A(index) = h * 1./sqrt(abs(sMatrix(index)-tMatrix(index)));
 
