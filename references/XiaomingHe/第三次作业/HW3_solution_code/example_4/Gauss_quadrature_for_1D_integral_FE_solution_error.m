function result=Gauss_quadrature_for_1D_integral_FE_solution_error(uh_local,accurate_function,vertices,Gauss_coefficient_local_1D,Gauss_point_local_1D,basis_type,derivative_degree)
%Xiaoming He, 10/08/2011.
%We will use "FE" to replace "finite element" in the comments.
%Use Gauss quadrature to numerically compute a norm error of FE solution on a local 1D element T.
%accurate_function: the accurate function in the error.
%When we take the L2 norm,accurate_function is the exact solution.
%When we take the H1 seminorm, accurate_function is the first derivative of the exact solution.
%vertices: the coordinates of the vertices of the triangular element T.
%uh_local: the values of the FE solution at the nodes of FE in the 1D element T.
%Gauss_coefficient_local_1D,Gauss_point_local_1D: the Gauss coefficients and Gauss points on the local interval.
%derivative_degree:the derivative degree of the FE solution.
%Gpn: the Gauss point number.

Gpn=length(Gauss_coefficient_local_1D);

result=0;
for i=1:Gpn
    result=result+Gauss_coefficient_local_1D(i)*(feval(accurate_function,Gauss_point_local_1D(i))-FE_solution_1D(Gauss_point_local_1D(i),uh_local,vertices,basis_type,derivative_degree))^2;
end