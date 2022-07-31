function r=exact_solution(x,y)
%Xiaoming He, 07/05/2009.

r=x.*y.*(1-x/2).*(1-y).*exp(x+y);