function r=Gauss_quadrature_for_volume_integral_trial_test_triangle(coefficient_function_name,Gauss_coefficient_local,Gauss_point_local,vertices,trial_basis_type,trial_basis_index,trial_derivative_degree_x,trial_derivative_degree_y,test_basis_type,test_basis_index,test_derivative_degree_x,test_derivative_degree_y)
%Xiaoming He, 07/01/2009.
%Use Gauss quadrature to compute a volume integral on a local triangular element T for a matrix.
%We will use "FE" to replace "finite element" in the comments.
%The integrand of the volume integral must be in the following format:
%a coefficient function * a trial FE basis function(or its derivatives) * a test FE basis function (or its derivatives).
%coefficient_function_name: the coefficient function of the integrand.
%Gauss_coefficient_local,Gauss_point_local:the Gauss coefficients and Gauss points on the triangular element T.
%vertices: the coordinates of all vertices of the triangular element T.
%trial_basis_type:the type of the trial FE basis function.
%trial_basis_type=1:2D linear FE.  
%trial_basis_type=2:2D Lagrange quadratic FE.
%trial_basis_index: the index of trial FE basis function to specify which trial FE basis function we want to use.
%trial_derivative_degree_x:the derivative degree of the trial FE basis function with respect to x.
%trial_derivative_degree_y:the derivative degree of the trial FE basis function with respect to y.
%test_basis_type:the type of the test FE basis function.
%test_basis_type=1:2D linear FE.  
%test_basis_type=2:2D Lagrange quadratic FE.
%test_basis_index: the index of test FE basis function to specify which test FE basis function we want to use.
%test_derivative_degree_x:the derivative degree of the test FE basis function with respect to x.
%test_derivative_degree_y:the derivative degree of the test FE basis function with respect to y.

%Gpn: the number of the Gauss points of the Gauss quadrature we are using.

Gpn=length(Gauss_coefficient_local);
r=0;
for i=1:Gpn
     r=r+Gauss_coefficient_local(i)*feval(coefficient_function_name,Gauss_point_local(i,1),Gauss_point_local(i,2))*triangular_local_basis(Gauss_point_local(i,1),Gauss_point_local(i,2),vertices,trial_basis_type,trial_basis_index,trial_derivative_degree_x,trial_derivative_degree_y)*triangular_local_basis(Gauss_point_local(i,1),Gauss_point_local(i,2),vertices,test_basis_type,test_basis_index,test_derivative_degree_x,test_derivative_degree_y);
end