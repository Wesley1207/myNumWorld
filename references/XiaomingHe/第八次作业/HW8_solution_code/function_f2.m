function r=function_f2(x,y)
%Xiaoming He, 10/02/2011.
lamda=function_lamda(x,y);
mu=function_mu(x,y);
r=-(lamda+2*mu)*(2*x.*(x-1))-(lamda+mu)*(pi^2*cos(pi*x).*cos(pi*y))-mu*(2*y.*(y-1));                                                                                                                       