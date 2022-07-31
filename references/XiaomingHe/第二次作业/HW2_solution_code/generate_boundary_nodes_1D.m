function boundary_nodes=generate_boundary_nodes_1D(N_basis)
%Xiaoming He, 10/08/2011.
%Generate the information matrix for boundary nodes.
%This function needs to be modified for different boundary conditons.
%This function needs to be modified if we change the index of nodes and elements in "generate_M_T_1D.m".
%We will use "FE" to replace "finite element" in the comments.
%N_basis:The N for the FE basis functions,not the partition.
%N is the number of the sub-intervals.



%boundary_nodes(1,k): specifiy the type of the kth boundary node.
%boundary_nodes(1,k)=-1: Dirichlet boundary node;
%boundary_nodes(1,k)=-2: Neumann boundary node;
%boundary_nodes(1,k)=-3: Robin boundary node. 
%boundary_nodes(2,k): global index of the kth boundary node among all nodes of FE. 
%                     That is, the index of FE is used here.
%boundary_nodes(3,k): The normal direction of the kth boundary node.


%Information matrix for boundary nodes. It uses the index of FE, not the index of partition.


%The following boundary condition may change for different problems.
%All Dirichlet boundary nodes.
boundary_nodes(1,1)=-1;
boundary_nodes(2,1)=1;
boundary_nodes(3,1)=-1;
boundary_nodes(1,2)=-1;
boundary_nodes(2,2)=N_basis+1;
boundary_nodes(3,2)=1;


