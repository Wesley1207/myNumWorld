function result=FE_solution_error_triangle_index(uh,accurate_function,left,right,bottom,top,h_partition,basis_index,basis_type,derivative_degree_x,derivative_degree_y,Gauss_point_number)
%Xiaoming He, 09/04/2011.
%We will use "FE" to replace "finite element" in the comments.
%Numerically compute a norm error of FE solution on the whole domain [left,right]*[bottom,top].
%uh: the values of the FE solution at all nodes of FE in the whole domain. These values are in 1D index of nodes of FE.
%accurate_function: the accurate function in the error.
%When we take the L2 norm,accurate_function is the exact solution.
%When we take the H1 seminorm, accurate_function is the first derivative of the exact solution.
%h_partition: the stepsize of the partition.
%basis_type: the type of the FE.
%basis_type=0:2D constant FE.
%basis_type=1:2D Lagrange linear FE.
%basis_type=2:2D Lagrange quadratic FE.
%basis_type=10:2D Crouzeix-Raviart FE.
%derivative_degree_x:the derivative degree of the FE solution with respect to x.
%derivative_degree_y:the derivative degree of the FE solution with respect to y.
%Gauss_point_number: the number of the Gauss points of the Gauss quadrature we want to use.

%N1_partition,N2_partition:The N1 and N2 for the partition,not the FE basis functions.
%N1 is the number of the sub-intervals of the partition in x-direction.
%N2 is the number of the sub-intervals of the partition in y-direction.
%vertices: the coordinates of the vertices of a triangular element.
%uh_local: the values of the FE solution at the nodes of FE in a triangular
%element.
N1_partition=(right-left)/h_partition(1);
N2_partition=(top-bottom)/h_partition(2);
number_of_elements=2*N1_partition*N2_partition;

[M_partition,T_partition]=generate_M_T_triangle(left,right,bottom,top,h_partition,1);

if basis_type==1
    M_basis=M_partition;
    T_basis=T_partition;
else
    [M_basis,T_basis]=generate_M_T_triangle(left,right,bottom,top,h_partition,basis_type);
end 

[Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle]=generate_Gauss_reference_triangle(Gauss_point_number);

result=0;
%Go through all elements and accumulate the error on them.
for n=1:number_of_elements
    vertices=M_partition(:,T_partition(:,n));
    uh_local=uh(T_basis(:,n));
    result=result+Gauss_quad_for_volume_integral_FE_solution_error_triangle_index(uh_local,accurate_function,vertices,basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,derivative_degree_x,derivative_degree_y);
end
result=sqrt(result);