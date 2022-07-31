function result=Gauss_quadrature_for_1D_integral_test(coefficient_function_name,Gauss_coefficient_local_1D,Gauss_point_local_1D,test_vertices,test_basis_type,test_basis_index,test_derivative_degree)
%Xiaoming He, 10/08/2011.
%Use Gauss quadrature to compute a 1D integral for a vector.
%We will use "FE" to replace "finite element" in the comments. 
%The integrand of the 1D integral must be in the following format:
%a coefficient function * a test FE basis function (or its derivatives).
%coefficient_function_name: the coefficient function of the integrand.
%Gauss_coefficient_reference_1D,Gauss_point_reference_1D:the Gauss coefficients and Gauss points on the reference interval [-1,1].
%end_point_1,end_point_2:the coordinates of the end points of the edge on which we are computing the 1D integral.
%test_vertices: the coordinates of all vertices of the triangular element for test functions.
%test_basis_type:the type of the test FE basis function.
%test_basis_index: the index of test FE basis function to specify which test FE basis function we want to use.
%test_derivative_degree:the derivative degree of the test FE basis function.

%Gpn: the number of the Gauss points of the Gauss formula we are using
%Gauss_coefficient_local_1D,Gauss_point_local_1D: the Gauss coefficients and Gauss points on the local interval.

Gpn=length(Gauss_coefficient_local_1D);

result=0;
for i=1:Gpn
     result=result+Gauss_coefficient_local_1D(i)*feval(coefficient_function_name,Gauss_point_local_1D(i))*local_basis_1D(Gauss_point_local_1D(i),test_vertices,test_basis_type,test_basis_index,test_derivative_degree);
end   