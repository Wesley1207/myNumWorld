function [A,b]=treat_Dirichlet_boundary_Stokes(Dirichlet_boundary_function_name_u1,Dirichlet_boundary_function_name_u2,A,b,boundary_nodes,M_basis_u,number_of_FE_nodes_u)
%Xiaoming He, 07/11/2009.
%Deal with Dirichlet boundary nodes.
%We will use "FE" to replace "finite element" in the comments.
%Dirichelet_boundary_fucntion_name_u1: the name of the Dirichelet boundary function for u in the normal direction(or u1).
%Dirichelet_boundary_fucntion_name_u2: the name of the Dirichelet boundary function for u in the tangential direction(or u2).
%boundary_nodes(1,k): specifiy the type of the kth boundary node for the normal direction(or u1).
%boundary_nodes(1,k)=-1: Dirichlet boundary node in the normal direction(or u1);
%boundary_nodes(1,k)=-2: Stress boundary node in the normal direction(or u1);
%boundary_nodes(1,k)=-3: Robin boundary node in the normal direction(or u1). 
%boundary_nodes(2,k): specifiy the type of the kth boundary node for the tangential direction(or u2).
%boundary_nodes(2,k)=-1: Dirichlet boundary node in the tangential direction(or u2);
%boundary_nodes(2,k)=-2: Stress boundary node in the tangential direction(or u2);
%boundary_nodes(2,k)=-3: Robin boundary node in the tangential direction(or u2).
%The intersection node between Dirichlet boundary and other boundaries is a Dirichlet boundary node.
%boundary_nodes(3,k): global index of the kth boundary node among all nodes of FE. 
%                     That is, the index of FE is used here.
%M_basis: store the coordinates of all the nodes for the FE,not the partition.
%"_u" is for the velocity vector function u.
%number_of_FE_nodes_u: the number of the FE nodes for u. 
%See "generate_M_T_triangular.m" for more explanation about M_basis.
%More explanation is in my "Notes for tool box of standard triangular FE" section 1-6.

%nbn: the total number of all the boundary nodes of FE.

nbn=size(boundary_nodes,2);

%Check all boundary nodes of FE.
for k=1:nbn

%If the ith FE node X_i is a Dirichlet boundary node in the normal direction(or u1),
%then we reset the ith equation in the linear sysytem to be 1*u1_i=u1(X_i).
%Here u1_i is the unknown associated with the node X_i for u1 and u1(X_i)=Dirichelet_boundry_function_name_u1(X_i).
%Also i=boundary_nodes(3,k) if boundary_nodes(1,k)==-1.
%If the ith FE node X_i is a Dirichlet boundary node in the tangential direction(or u1),
%then we reset the (number_of_FE_nodes_u+i)th equation in the linear sysytem to be 1*u2_i=u2(X_i).
%Here u2_i is the unknown associated with the node X_i for u2 and u2(X_i)=Dirichelet_boundry_function_name_u2(X_i).
%Again i=boundary_nodes(3,k) if boundary_nodes(2,k)==-1.

    if boundary_nodes(1,k)==-1 
        i=boundary_nodes(3,k);
        A(i,:)=0;
        A(i,i)=1;
        b(i,1)=feval(Dirichlet_boundary_function_name_u1,M_basis_u(1,i),M_basis_u(2,i));
    end

    
    if boundary_nodes(2,k)==-1 
        i=boundary_nodes(3,k);
        u2_index=number_of_FE_nodes_u+i;
        A(u2_index,:)=0;
        A(u2_index,u2_index)=1;
        b(u2_index,1)=feval(Dirichlet_boundary_function_name_u2,M_basis_u(1,i),M_basis_u(2,i));
    end
end
