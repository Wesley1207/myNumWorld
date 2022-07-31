function r=poisson_solver_triangle(left,right,bottom,top,h_partition,basis_type)
%Xiaoming He, 07/04/2009.
%Finite element solver for Poisson equation on a triangular mesh.
%We will use "FE" to replace "finite element" in the comments.
%The trial FE functions and test FE functions need to be the same.
%So far the code can only handle 2D linear FE and 2D Lagrange quadratic FE.
%For other types of FE, we need to add the corresponding information to code.
%The problem domain is [left,right]*[bottom,top].
%h_partition is the step size of the partition.
%basis_type:the type of the FE basis function.
%basis_type=1:2D linear FE.  
%basis_type=2:2D Lagrange quadratic FE.

%N1_basis,N2_basis:The N1 and N2 for the FE basis functions,not the partition.
%N1_partition,N2_partition:The N1 and N2 for the partition,not the FE basis functions.
%N1 is the number of the sub-intervals of the partition in x-direction.
%N2 is the number of the sub-intervals of the partition in y-direction.
%M_partition,T_partition, M_basis,T_basis: see the note in "generate_M_T_triangular.m".
%function_a: the coefficient function on the left side of the poisson equation.
%funciton_f: the right hand side function of the poisson equation.
%function_g: the Dirichelet boundary function in my notes "Notes for tool box of standard triangular FE" section 1-1.
%function_q_tilde: the name of the Neumann boundary function q(x,y) when p(x,y)=0 in my notes "Notes for tool box of standard triangular FE" section 1-1.
%function_q: the name of the Neumann boundary function q(x,y) when p(x,y) is nonzero in my notes "Notes for tool box of standard triangular FE" section 1-1.
%function_p: the name of the Robin coefficient function p(x,y) in my notes "Notes for tool box of standard triangular FE" section 1-1.

%h_basis is the step size for the finite element nodes, not the partition.

N1_partition=(right-left)/h_partition(1);
N2_partition=(top-bottom)/h_partition(2);

if basis_type==2
    N1_basis=N1_partition*2;
    N2_basis=N2_partition*2;
elseif basis_type==1
    N1_basis=N1_partition;
    N2_basis=N2_partition;
end

%Mesh information for partition and finite element basis functions.
[M_partition,T_partition]=generate_M_T_triangle(left,right,bottom,top,h_partition,1);

if basis_type==2
    [M_basis,T_basis]=generate_M_T_triangle(left,right,bottom,top,h_partition,2);
elseif basis_type==1
    M_basis=M_partition;
    T_basis=T_partition;
end 


%Guass quadrature's points and weights on the refenrece triangle and reference interval.
[Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle]=generate_Gauss_reference_triangle(9);
%The following line is necessary if we need to deal with Neumann or Robin boundary condition. 
%Otherwise, we can comment this line to save time and memory.
[Gauss_coefficient_reference_1D,Gauss_point_reference_1D]=generate_Gauss_reference_1D(4);

%Assemble the stiffness matrix.
number_of_elements=2*N1_partition*N2_partition;
matrix_size=[(N1_basis+1)*(N2_basis+1) (N1_basis+1)*(N2_basis+1)];
if basis_type==2
    number_of_trial_local_basis=6;
    number_of_test_local_basis=6;
elseif basis_type==1
    number_of_trial_local_basis=3;
    number_of_test_local_basis=3;
end
A1=assemble_matrix_from_volume_integral_triangle('function_a',M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,1,0,basis_type,1,0);
A2=assemble_matrix_from_volume_integral_triangle('function_a',M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,1,basis_type,0,1);
A=A1+A2;

%Assemble the load vector.
vector_size=(N1_basis+1)*(N2_basis+1);
b=assemble_vector_from_volume_integral_triangle('function_f',M_partition,T_partition,T_basis,number_of_test_local_basis,number_of_elements,vector_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,0);

%Get the information matrices for boundary nodes and boundary edges.
[boundary_nodes,boundary_edges]=generate_boundary_nodes_edges(N1_basis,N2_basis,N1_partition,N2_partition);

%Deal with Neumann boundary condition. If we don't have Neumann boundary condition at all, we can comment the following line to save time and memory.
b=treat_Neumann_boundary_triangle('function_q_tilde',b,boundary_edges,M_partition,T_partition,T_basis,number_of_test_local_basis,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,basis_type,0,0);

%Deal with Robin boundary condition. If we don't have Robin boundary condition at all, we can comment the following line to save time and memory.
[A,b]=treat_Robin_boundary_triangle('function_q','function_p',A,b,boundary_edges,M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,basis_type,0,0,basis_type,0,0);

%Deal with Dirichlet boundary condition.
[A,b]=treat_Dirichlet_boundary_triangle('function_g',A,b,boundary_nodes,M_basis);

%Compute the numerical solution
r=A\b;

%Transfer the 1D solution into 2D solution and compute the maximum error at all nodes.
if basis_type==2
    h_basis=h_partition/2;
elseif basis_type==1
    h_basis=h_partition;
end
[solution_2D,maxerror]=get_2D_solution_and_maximum_error(r,N1_basis,N2_basis,left,bottom,h_basis);
maximum_error_at_all_nodes_of_FE=maxerror





