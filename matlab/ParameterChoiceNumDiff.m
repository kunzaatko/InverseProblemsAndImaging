%Forelæsning

NumericalDifferentiation;
%Provides A, yvec, xvec

%add noise 
delta = 0.1;
n = randn(N,1);
ydelta = yvec + delta*n/norm(n);

%Reg parameter
j=-7:.2:0;
alpha = 10.^(j);
M = length(alpha);
 xalpha =zeros(N,M);

 F1 = zeros(M,1);
F2 = zeros(M,1);
% Find Tikhonov solution
 for j = 1:M
     alphaj = alpha(j);
     Aalpha  = [A;sqrt(alphaj)*eye(N)];
     yext = [ydelta;zeros(N,1)];
     xalpha(:,j) = Aalpha \ yext;
     F1(j)=norm(A*xalpha(:,j)-ydelta);
     F2(j) = norm(xalpha(:,j));
 end;

figure(3)
loglog(alpha,F1)
hold on
loglog(alpha,delta*ones(M,1),'b--','linewidth', 2)
loglog(alpha(23),F1(23),'*')
xlabel('\alpha')
ylabel('||Kx_\alpha-y^\delta||')
hold off

figure(4)
loglog(F1,F2)
xlabel('||Kx_\alpha-y^\delta||')
ylabel('||x_\alpha||')
hold on
loglog(F1(22),F2(22),'*')
