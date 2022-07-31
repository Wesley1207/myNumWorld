function r=Gauss_quadrature_for_line_integral_test_triangle(coefficient_function_name,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,end_point_1,end_point_2,test_vertices,test_basis_type,test_basis_index,test_derivative_degree_x,test_derivative_degree_y)
%Xiaoming He, 07/04/2009.
%Use Gauss quadrature to compute a line integral on an edge for a vector.
%We will use "FE" to replace "finite element" in the comments. 
%The integrand of the volume integral must be in the following format:
%a coefficient function * a test FE basis function (or its derivatives).
%coefficient_function_name: the coefficient function of the integrand.
%Gauss_coefficient_reference_1D,Gauss_point_reference_1D:the Gauss coefficients and Gauss points on the reference interval [-1,1].
%end_point_1,end_point_2:the coordinates of the end points of the edge on which we are computing the line integral.
%test_vertices: the coordinates of all vertices of the triangular element T for test functions.
%test_basis_type:the type of the test FE basis function.
%test_basis_type=1:2D linear FE.  
%test_basis_type=2:2D Lagrange quadratic FE.
%test_basis_index: the index of test FE basis function to specify which test FE basis function we want to use.
%test_derivative_degree_x:the derivative degree of the test FE basis function with respect to x.
%test_derivative_degree_y:the derivative degree of the test FE basis function with respect to y.

%Gpn: the number of the Gauss points of the Gauss formula we are using
%Gauss_coefficient_local_1D,Gauss_point_local_1D: the Gauss coefficients and Gauss points on the local interval.

Gpn=length(Gauss_coefficient_reference_1D);
r=0;

if end_point_1(2)==end_point_2(2)
%The edge is horizontal.

    lower_bound=min(end_point_1(1),end_point_2(1));
    upper_bound=max(end_point_1(1),end_point_2(1));
    [Gauss_coefficient_local_1D,Gauss_point_local_1D]=generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound);
    for i=1:Gpn
         r=r+Gauss_coefficient_local_1D(i)*feval(coefficient_function_name,Gauss_point_local_1D(i),end_point_1(2))*triangular_local_basis(Gauss_point_local_1D(i),end_point_1(2),test_vertices,test_basis_type,test_basis_index,test_derivative_degree_x,test_derivative_degree_y);
    end    

elseif end_point_1(1)==end_point_2(1)
%The edge is vertical.

    lower_bound=min(end_point_1(2),end_point_2(2));
    upper_bound=max(end_point_1(2),end_point_2(2));
    [Gauss_coefficient_local_1D,Gauss_point_local_1D]=generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound);
    for i=1:Gpn
         r=r+Gauss_coefficient_local_1D(i)*feval(coefficient_function_name,end_point_1(1),Gauss_point_local_1D(i))*triangular_local_basis(end_point_1(1),Gauss_point_local_1D(i),test_vertices,test_basis_type,test_basis_index,test_derivative_degree_x,test_derivative_degree_y);
    end 
    
else
%The slope of the edge is in (0,infinity).

    lower_bound=min(end_point_1(1),end_point_2(1));
    upper_bound=max(end_point_1(1),end_point_2(1));
    [Gauss_coefficient_local_1D,Gauss_point_local_1D]=generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound);
    slope=(end_point_2(2)-end_point_1(2))/(end_point_2(1)-end_point_1(1));
    Jacobi=sqrt(1+slope^2);
    for i=1:Gpn
         x=Gauss_point_local_1D(i);
         y=slope*(x-end_point_1(1))+end_point_1(2);
         r=r+Gauss_coefficient_local_1D(i)*Jacobi*feval(coefficient_function_name,x,y)*triangular_local_basis(x,y,test_vertices,test_basis_type,test_basis_index,test_derivative_degree_x,test_derivative_degree_y);
    end 
    
end