function [boundary_nodes,boundary_edges]=generate_boundary_nodes_edges(N1_basis,N2_basis,N1_partition,N2_partition)
%Xiaoming He, 07/02/2009.
%Generate the information matrices for boundary nodes and boundary edges.
%This function needs to be modified for different boundary conditons.
%Here, this function is for pure Dirichlet boundary condition.
%This function needs to be modified if we change the index of nodes and elements in "generate_M_T_triangle.m".
%We will use "FE" to replace "finite element" in the comments.
%N1_basis,N2_basis:The N1 and N2 for the FE basis functions,not the partition.
%N1 is the number of the sub-intervals of the partition in x-direction.
%N2 is the number of the sub-intervals of the partition in y-direction.


%boundary_nodes(1,k): specifiy the type of the kth boundary node.
%boundary_nodes(1,k)=-1: Dirichlet boundary node;
%boundary_nodes(1,k)=-2: Neumann boundary node;
%boundary_nodes(1,k)=-3: Robin boundary node. 
%boundary_nodes(1,k)=-100: Don't treat this node as a boundary node.
%boundary_nodes(2,k): global index of the kth boundary node among all nodes of FE. 
%                     That is, the index of FE is used here.
%boundary_edges(1,k): specifiy the type of the kth boundary edge.
%boundary_edges(1,k)=-1: Dirichlet boundary edge;
%boundary_edges(1,k)=-2: Neumann boundary edge;
%boundary_edges(1,k)=-3: Robin boundary edge. 
%boundary_edges(1,k)=-100: Don't treat this edge as a boundary edge.
%boundary_edges(2,k): index of the element which contains the kth boundary edge.
%boundary_edges(3:4,k): indices of the two end points of the kth boundary edge among all grid points, not the nodes of FE.
%                       That is, the index of partition is used here.
%More explanation is in my "Notes for tool box of standard triangular FE" section 1-6.

%nbn: the total number of all the boundary nodes of FE.
%nbe: the total number of all the boundary edges.


%Information matrix for boundary nodes. It uses the index of FE, not the index of partition.

nbn=2*(N1_basis+N2_basis);
boundary_nodes=zeros(2,nbn);

%The following boundary condition may change for different problems.
%All Dirichlet boundary nodes.
boundary_nodes(1,:)=-1;

%The index in the following is associated with the index in "generate_M_T_triangle.m".
%bottom boundary nodes.
for k=1:N1_basis
    boundary_nodes(2,k)=(k-1)*(N2_basis+1)+1;
end

%right boundary nodes.
for k=N1_basis+1:N1_basis+N2_basis
    boundary_nodes(2,k)=N1_basis*(N2_basis+1)+k-N1_basis;
end

%top boundary nodes.
for k=N1_basis+N2_basis+1:2*N1_basis+N2_basis
    boundary_nodes(2,k)=(2*N1_basis+N2_basis+2-k)*(N2_basis+1);
end

%left boundary nodes.
for k=2*N1_basis+N2_basis+1:nbn
    boundary_nodes(2,k)=2*N1_basis+2*N2_basis+2-k;
end





%Information matrix for boundary edges. It uses the index of partition, not the index of FE.

nbe=2*(N1_partition+N2_partition);
boundary_edges=zeros(4,nbe);


%The following boundary condition may change for different problems.
%All Dirichlet boundary edges.
boundary_edges(1,:)=-1;

%The index in the following is associated with the index in "generate_M_T_triangle.m".
%bottom boundary edges.
for k=1:N1_partition
    boundary_edges(2,k)=(k-1)*2*N2_partition+1;
    boundary_edges(3,k)=(k-1)*(N2_partition+1)+1;
    boundary_edges(4,k)=k*(N2_partition+1)+1;
end

%right boundary edges.
for k=N1_partition+1:N1_partition+N2_partition
    boundary_edges(2,k)=(N1_partition-1)*2*N2_partition+2*(k-N1_partition);
    boundary_edges(3,k)=N1_partition*(N2_partition+1)+k-N1_partition;
    boundary_edges(4,k)=N1_partition*(N2_partition+1)+k-N1_partition+1;
end

%top boundary edges.
for k=N1_partition+N2_partition+1:2*N1_partition+N2_partition
    boundary_edges(2,k)=(2*N1_partition+N2_partition+1-k)*2*N2_partition;
    boundary_edges(3,k)=(2*N1_partition+N2_partition+2-k)*(N2_partition+1);
    boundary_edges(4,k)=(2*N1_partition+N2_partition+1-k)*(N2_partition+1);
end

%left boundary edges.
for k=2*N1_partition+N2_partition+1:nbe
    boundary_edges(2,k)=2*(2*N1_partition+2*N2_partition+1-k)-1;
    boundary_edges(3,k)=2*N1_partition+2*N2_partition+2-k;
    boundary_edges(4,k)=2*N1_partition+2*N2_partition+1-k;
end



