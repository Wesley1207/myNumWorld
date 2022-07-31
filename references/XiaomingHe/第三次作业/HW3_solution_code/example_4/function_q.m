function result=function_q(x)
%Xiaoming He, 10/08/2011.

left=0;

if x==left
    result=function_a(x).*exact_solution_derivative(x)+function_p(x).*exact_solution(x);
end