function result=function_q_tilde(x)
%Xiaoming He, 10/08/2011.
right=1;

if x==right
    result=function_a(x).*exact_solution_derivative(x);
end