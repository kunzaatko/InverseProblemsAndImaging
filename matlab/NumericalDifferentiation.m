% Numerical differentiation as an inverse problem
% Course 2624 - week 1
% Problem: y'(s) = x(s), y(0) = 0; given by y(s) = int(x(t),t=0..s) = Kx(s)
% Approximation of forward K by matrix from quadrature rule
% Solve inverse problem: Compute x from y
%Provides matrix A and right hand side yvec
% Kim Knudsen 2024

clear
close all

%True solution x and data y
y = @(s)  s.*exp(-s);
x = @(s)  exp(-s).*(1-s);

% Initialization of grid
N = 200;
h = 1/N;
s = (0:h:1-h)';

% Numerical integration via quadrature: Matrix A implements left point
% quadrature rule
% NB y is computed on shifted grid
xvec = x(s);
A = tril(ones(N))*h;
yvec = A * xvec;


% Inspect functions
figure(1)
hold on
plot(s,xvec);
legend('x(s)');
hold off

figure(2)
hold on
plot(s+h,yvec);
plot(s+h,y(s+h));
legend('y(s)');
hold off

%pause

% Add noise
noise = 0.01*randn(N,1);
yvecnoise = yvec + noise;
noiselevel = norm(noise)/norm(yvec)

% Solve naively the inverse problem
xvecrecon = A\yvecnoise;

% Calculate relative error in result
relerror = norm(xvec-xvecrecon)/norm(xvec)

% Inspect results
figure(2)
hold on
plot(s+h,yvecnoise)
legend('ytrue','y(s)','ynoise(s)')
hold off

figure(1)
hold on
plot(s,xvecrecon)
legend('x(s)','xnoise(s)')
hold off
