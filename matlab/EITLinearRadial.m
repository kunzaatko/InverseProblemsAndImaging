% We setup the linearized EIT problem for radial profile in unit ball
% Week 11 in 02624
% Kim Knudsen 2019-21

close all
clear

%load precomputed eigenvalues for difference ND operator.
%For instance using computeEigVal
load b;
b = lambdaNdiff;


%Set up matrix
N = length(b);
Nvek = (1:N)';

%Number of discrete points on r axis
M = 100;
dr= 1/(M);
rvec = dr:dr:1;
rmat = ones(N,1)*rvec ;
Nmat = Nvek * ones(1,M);

%This matrix approximates forward problem for linearized EIT
A = -2*rmat.^(2*Nmat-1)*dr;

%The linear discrete problem is now Ax=b
%Need to solve it via regularization
