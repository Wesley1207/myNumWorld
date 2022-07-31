function b=treat_Neumann_boundary_triangle(Neumann_boundary_function_name,b,boundary_edges,M_partition,T_partition,T_basis_test,number_of_test_local_basis,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,test_basis_type,test_derivative_degree_x,test_derivative_degree_y)
%Xiaoming He, 07/04/2009.
%Deal with Neumann boundary edges.
%We will use "FE" to replace "finite element" in the comments.
%Neumann_boundary_fucntion_name: the name of the Neumann boundary function q(x,y) in my notes "Notes for tool box of standard triangular FE" section 1-1.
%b: the vector affected by the Neumann boundary condition.
%boundary_edges(1,k): specifiy the type of the kth boundary edge.
%boundary_edges(1,k)=-1: Dirichlet boundary edge;
%boundary_edges(1,k)=-2: Neumann boundary edge;
%boundary_edges(1,k)=-3: Robin boundary edge. 
%boundary_edges(2,k): index of the element which contains the kth boundary edge.
%boundary_edges(3:4,k): indices of the two end points of the kth boundary edge among all grid points, not the nodes of FE.
%                       That is, the index of partition is used here.
%M_partition: store the coordinates of all the grid points of the partition,not FE.
%T_partition: store the global indices of the grid points of every element for the partition,not FE.
%T_basis: store the global indices of the nodes of every element for FE,not the partition.
%T_basis_test: T_basis for the test basis function.
%The explanation for M_partition,T_partition,T_basis is in generate_M_T_triangular.m.
%number_of_test_local_basis: the number of local FE basis functions for the test function in a local element.
%Gauss_coefficient_reference_1D,Gauss_point_reference_1D:the Gauss coefficients and Gauss points on the reference interval [-1,1].
%test_basis_type:the type of the test FE basis function.
%test_basis_type=1:2D linear FE.  
%test_basis_type=2:2D Lagrange quadratic FE.
%test_basis_index: the index of test FE basis function to specify which test FE basis function we want to use.
%test_derivative_degree_x:the derivative degree of the test FE basis function with respect to x.
%test_derivative_degree_y:the derivative degree of the test FE basis function with respect to y.
%More explanation is in my "Notes for tool box of standard triangular FE" section 1-6.

%nbe: the total number of all the boundary edges.

nbe=size(boundary_edges,2);

%Check all boundary edges.
for k=1:nbe

%If the kth boundary edge is a Neumann boundary edge,then we add the corresponding line integral to b.

    if boundary_edges(1,k)==-2 
        
        element_index=boundary_edges(2,k);
        vertices=M_partition(:,T_partition(:,element_index));
        end_point_1=M_partition(:,boundary_edges(3,k));
        end_point_2=M_partition(:,boundary_edges(4,k));
        for beta=1:number_of_test_local_basis
            temp=Gauss_quadrature_for_line_integral_test_triangle(Neumann_boundary_function_name,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,end_point_1,end_point_2,vertices,test_basis_type,beta,test_derivative_degree_x,test_derivative_degree_y);
            b(T_basis_test(beta,element_index),1)=b(T_basis_test(beta,element_index),1)+temp;
        end
        
    end

end