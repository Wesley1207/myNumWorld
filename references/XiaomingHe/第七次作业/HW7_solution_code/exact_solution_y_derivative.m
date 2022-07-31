function r=exact_solution_y_derivative(t,x,y)
%Xiaoming He, 07/17/2009.

r=(2-pi*sin(pi*x)).*(-1+pi*sin(pi*(1-y))).*cos(2*pi*t);