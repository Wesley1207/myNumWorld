function r=function_f(x,y)
%Xiaoming He, 07/05/2009.

r=-y.*(1-y).*(1-x-x.*x/2).*exp(x+y)-x.*(1-x/2).*(-3*y-y.*y).*exp(x+y);                                                                                                                        