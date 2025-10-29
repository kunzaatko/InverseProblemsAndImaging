NumericalDifferentiation;
%Provides A matrix, xvec, yvec, yvecnoise

%Tikhonov
alpha = 10^(-2);
Aalpha  = [A;sqrt(alpha)*eye(N)];
yext = [yvecnoise;zeros(N,1)];
xalpha = Aalpha \ yext;

close all
 figure
 plot(s,xvec);
 hold on;
 plot(s,xalpha);
 %plot(s,xhat);
 legend('xtrue', 'alpha = 10^{-2}')
 title('Tikhonov for num diff')
 xlabel('s')
 ylabel('x(s)')

%Landweber iteration
x0 = zeros(1,N)';
omega = 0.9/norm(A,2)^2;

K=200;

xland= zeros(N,K);
relerror = zeros(1,K);
relerror(1) = 1;

for k=2:K
    xland(:,k) = xland(:,k-1) + omega*A'*(-A*xland(:,k-1) + yvecnoise);
    relerror(k) = norm(xland(:,k)-xvec)/norm(xvec);
end;
figure(2)
hold on
plot(s,xvec)
plot(s,xland(:,1:20:K))
 %legend('xtrue', 'alpha = 10^{-2}')
 title('Landweber for num diff')
 xlabel('s')
 ylabel('x(s)')
 hold off

 figure(3) 
 semilogy(2:K,relerror(2:K));
 title('Relative error')
 xlabel('k')
 ylabel('E(k)')

 
%cgls
[X,rho,eta] = cgls(A,yvecnoise,5,0);
figure(4)
hold on;
plot(s,xvec,'*')
plot(s,X);
xlabel('s')
 ylabel('x(s)')
 title('cgls for num diff')