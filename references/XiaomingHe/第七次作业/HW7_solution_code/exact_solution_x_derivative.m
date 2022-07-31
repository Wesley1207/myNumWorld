function r=exact_solution_x_derivative(t,x,y)
%Xiaoming He, 07/17/2009.

r=-pi^2*cos(pi*x).*(-y+cos(pi*(1-y))).*cos(2*pi*t);