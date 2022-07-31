function r=exact_solution_x_derivative(x,y)
%Xiaoming He, 07/05/2009.

r=(y.*(1-x/2).*(1-y)-0.5*x.*y.*(1-y)+x.*y.*(1-x/2).*(1-y)).*exp(x+y);