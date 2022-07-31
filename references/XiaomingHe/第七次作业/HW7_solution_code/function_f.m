function r=function_f(t,x,y)
%Xiaoming He, 07/17/2009.

r=(-pi^3*sin(pi*x).*(-y+cos(pi*(1-y)))-(2-pi*sin(pi*x)).*(-pi^2*cos(pi*(1-y)))).*cos(2*pi*t)-2*pi*(2-pi*sin(pi*x)).*(-y+cos(pi*(1-y))).*sin(2*pi*t);                                                                                                                        