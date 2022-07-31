function [M,T]=generate_M_T_1D(left,right,h_partition,basis_type)
%Xiaoming He, 10/08/2011.
%Generate the matrix M and T for a mesh of a 1D domain [left,right].
%We will use "FE" to replace "finite element" in the comments.
%h_partition: the step size of the partition.
%We will use "FE" to replace "finite element" in the comments.
%basis_type: the type of the FE.
%basis_type=101:1D linear FE.
%basis_type=101:1D quadratic FE.
%M stores the coordinates of all nodes for the type of finite element specified by "basis_type".
%M(i,j) is the ith coordinate of the jth node.
%In 1D, M has only one row.
%T stores the global indices of the nodes of every element for the type of finite element specified by "basis_type".
%T(i,j) stores the global index of the ith node in th jth element.

%Note: the grid points of the partition are the same as the nodes of 1D linear FE basis functions.
%Hence the index of the partition is actually the same as that of 1D linear FE, so are M and T.
%However, sometimes the grid points of the partition are different from the nodes of FE basis functions.
%For example,1D Lagrange quadratic FE has 3 nodes correpsonding to the 3 nodal local FE basis functions in each element, 
%but the partition has only 2 grid points for each element,i.e, the 2 endpoints.
%So,we may have two kinds of M and T, which are M_partition,T_partition and M_basis, T_basis.
%M_partition: store the coordinates of all the grid points of the partition,not FE.
%T_partition: store the global indices of the grid points of every element for the partition,not FE.
%M_basis: store the coordinates of all the nodes for the FE,not the partition.
%T_basis: store the global indices of the nodes of every element for FE,not the partition.
%For 1D linear FE, M_partition,T_partition are the same as M_basis,T_basis.
%For 1D Langrange quadratic FE, they are different. The following 4 rows are what they store for 1D Langrange quadratic FE.
%M_partition only stores the coordinates of all grid points,i.e., vertices of all elements.
%M_basis stores the coordinates of all the nodes of FE.
%T_partition only stores the global indices of the 2 endpoints for every element.
%T_basis stores the global indices of the all the 3 nodes of FE for every element.
%The global index for T_partition is different the global index for T_basis:
%the global index for T_partition only counts the vertices of all the element;
%the global index for T_basis counts all the nodes for 1D Lagrange quadratic FE.

%N is the number of the sub-intervals of the partition.
%tnp:total number of all the nodes of FE,including inner nodes and boundary nodes.

h=h_partition;

if basis_type==101

   N=(right-left)/h;
   M=zeros(1,N+1);
   T=zeros(2,N);

   for i=1:N+1
       M(1,i)=left+(i-1)*h;       
   end

   for i=1:N
       T(1,i)=i;    
       T(2,i)=i+1;
   end
   
elseif basis_type==102

   N=(right-left)/h;
   dh=h/2;
   dN=N*2;
   M=zeros(1,dN+1);
   T=zeros(2,dN);

   for i=1:dN+1
       M(1,i)=left+(i-1)*dh;       
   end

   for i=1:N
       T(1,i)=2*i-1;    
       T(2,i)=2*i+1;
       T(3,i)=2*i;
   end
   
end
