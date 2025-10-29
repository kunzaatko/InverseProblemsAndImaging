% Forward heat operator
% Course 02624
% Problem: Given initial heat distribution x, compute terminal heat distribution y. Spatial domain is (0,pi), 
% terminal time T>0, Dirichlet boundary conditions.
% Approximation of y on equidistant grid uses FFT 
% Kim Knudsen 2024

  

% Parameters
L = pi; % Length of the interval
N = 2^8; % Number of spatial points
ds = L / N; % Spatial step size
s = 0:ds:L; % Spatial grid

% Terminal time parameter
T = .1; 

% Initial condition
u0 = sin(4*s) ; % Example initial condition
x0 = u0;

% Fourier modes
k = [-N:N-1]; % Wave numbers
k = fftshift(k); %Shift frequencies to match FFT

% Odd extension to force the full Fourier series as a sine series
x = x0(2:end-1);
x_ext = [0,x,0,-flip(x)];

%Compute Fourier coeff of x
x_hat = fft(x_ext); % FFT of the solution
x_hatT = x_hat .* exp(-k.^2 * T); % Multiply each Fourier coefficient by the right exponential 

xend = ifft(x_hatT); % Inverse FFT to get back to physical space
y = real(xend(1:N+1));

% Plot the final solution
plot(s, x0);
hold on
plot(s, y);
hold off
xlabel('s');
ylabel('x and y');
legend('x','y');
title('Solution');


