tic
clc;
clear;
left = -1;
right=1;
bottom=-1;
top=1;
h_partition=[1/64,1/64];
basis_type=2;

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

toc
