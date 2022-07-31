function r=function_f1(x,y)
%Xiaoming He, 10/02/2011.
lamda=function_lamda(x,y);
mu=function_mu(x,y);
r=-(lamda+2*mu)*(-pi^2*sin(pi*x).*sin(pi*y))-(lamda+mu)*((2*x-1).*(2*y-1))-mu*(-pi^2*sin(pi*x).*sin(pi*y));                                                                                                                        