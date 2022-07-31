function [A,b]=treat_Robin_boundary_1D(Neumann_boundary_function_name,Robin_boundary_function_name,A,b,boundary_nodes,M_basis)
%Xiaoming He, 10/08/2011.
%Deal with Robin boundary edges.
%We will use "FE" to replace "finite element" in the comments.
%A,b: the matrix and vector affected by the Robin boundary condition.
%boundary_nodes(1,k): specifiy the type of the kth boundary node.
%boundary_nodes(1,k)=-1: Dirichlet boundary node;
%boundary_nodes(1,k)=-2: Neumann boundary node;
%boundary_nodes(1,k)=-3: Robin boundary node. 
%boundary_nodes(2,k): global index of the kth boundary node among all nodes of FE. 
%                     That is, the index of FE is used here.%M_partition: store the coordinates of all the grid points of the partition,not FE.
%T_partition: store the global indices of the grid points of every element for the partition,not FE.
%T_basis: store the global indices of the nodes of every element for FE,not the partition.
%T_basis_trial: T_basis for the trial basis function.
%T_basis_test: T_basis for the test basis function.
%The explanation for M_partition,T_partition,T_basis is in generate_M_T_1D.m.
%number_of_trial_local_basis: the number of local FE basis functions for the trial function in a local element.
%number_of_test_local_basis: the number of local FE basis functions for the test function in a local element.
%Gauss_coefficient_reference_1D,Gauss_point_reference_1D:the Gauss coefficients and Gauss points on the reference interval [-1,1].
%trial_basis_type:the type of the trial FE basis function.
%trial_basis_index: the index of trial FE basis function to specify which trial FE basis function we want to use.
%trial_derivative_degree:the derivative degree of the trial FE basis function.
%test_basis_type:the type of the test FE basis function.
%test_basis_index: the index of test FE basis function to specify which test FE basis function we want to use.
%test_derivative_degree:the derivative degree of the test FE basis function.

%nbn: the total number of all the boundary nodes of FE.

nbn=size(boundary_nodes,2);

%Check all boundary nodes of FE.
for k=1:nbn

%If the kth boundary edge is a Robin boundary edge,then we add the corresponding values to A and b.

    if boundary_nodes(1,k)==-3 
        
        normal_direction=boundary_nodes(3,k);
        i=boundary_nodes(2,k);
        b(i,1)=b(i,1)+normal_direction*feval(Neumann_boundary_function_name,M_basis(i));
    
        A(i,i)=A(i,i)+normal_direction*feval(Robin_boundary_function_name,M_basis(i));
        
    end
    
end