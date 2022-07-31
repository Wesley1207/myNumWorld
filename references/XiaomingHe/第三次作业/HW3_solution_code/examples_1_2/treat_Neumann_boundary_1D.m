function b=treat_Neumann_boundary_1D(Neumann_boundary_function_name,b,boundary_nodes,M_basis)
%Xiaoming He, 10/08/2011.
%Deal with Neumann boundary edges.
%We will use "FE" to replace "finite element" in the comments.
%Neumann_boundary_fucntion_name: the name of the Neumann boundary function q(x,y) in my notes "Notes for tool box of standard triangular FE" section 1-1.
%b: the vector affected by the Neumann boundary condition.
%boundary_nodes(1,k): specifiy the type of the kth boundary node.
%boundary_nodes(1,k)=-1: Dirichlet boundary node;
%boundary_nodes(1,k)=-2: Neumann boundary node;
%boundary_nodes(1,k)=-3: Robin boundary node. 
%boundary_nodes(2,k): global index of the kth boundary node among all nodes of FE. 
%                     That is, the index of FE is used here.
%M_partition: store the coordinates of all the grid points of the partition,not FE.
%T_partition: store the global indices of the grid points of every element for the partition,not FE.
%T_basis: store the global indices of the nodes of every element for FE,not the partition.
%T_basis_test: T_basis for the test basis function.
%The explanation for M_partition,T_partition,T_basis is in generate_M_T_1D.m.
%number_of_test_local_basis: the number of local FE basis functions for the test function in a local element.
%test_basis_type:the type of the test FE basis function.
%test_basis_index: the index of test FE basis function to specify which test FE basis function we want to use.
%test_derivative_degree:the derivative degree of the test FE basis function.

%nbn: the total number of all the boundary nodes.

nbn=size(boundary_nodes,2);

%Check all boundary nodes of FE.
for k=1:nbn

%If the kth boundary edge is a Neumann boundary edge,then we add the corresponding values to b.

    if boundary_nodes(1,k)==-2 
        
        normal_direction=boundary_nodes(3,k);
        i=boundary_nodes(2,k);
        b(i,1)=b(i,1)+normal_direction*feval(Neumann_boundary_function_name,M_basis(i));
        
    end

end