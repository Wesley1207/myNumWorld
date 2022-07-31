function r=exact_solution_y_derivative(x,y)
%Xiaoming He, 07/05/2009.

r=(x.*(1-x/2).*(1-y)-x.*y.*(1-x/2)+x.*y.*(1-x/2).*(1-y)).*exp(x+y);