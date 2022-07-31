function [uh1,uh2]=Elasticity_solver_triangle(left,right,bottom,top,h_partition,basis_type)
%Xiaoming He, 10/02/2009.
%Finite element solver for Elasticity equation on a triangular mesh.
%We will use "FE" to replace "finite element" in the comments.
%The problem domain is [left,right]*[bottom,top].
%h_partition is the step size of the partition.

%N1_basis,N2_basis:The N1 and N2 for the FE basis functions,not the partition.
%N1_partition,N2_partition:The N1 and N2 for the partition,not the FE basis functions.
%N1 is the number of the sub-intervals of the partition in x-direction.
%N2 is the number of the sub-intervals of the partition in y-direction.
%M_partition,T_partition, M_basis,T_basis: see the note in "generate_M_T_triangular,m".
%"_u" is for the velocity vector function u.
%function_lamda, function_mu: the two coefficient functions in Elasticity equation.
%funciton_f1,funciton_f2: the right hand side functions of the Stokes equation.
%function_g1,function_g2: the Dirichelet boundary functions for (u1,u2) in my notes "Notes for tool box of standard triangular FE" section 2-3-2.
%h_basis is the step size for the finite element nodes, not the partition.
%More explanation about the structure of this driver is in my notes "Notes for tool box of standard triangular FE" section 2 and 7.

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

if basis_type==1
    M_basis=M_partition;
    T_basis=T_partition;
else
    [M_basis,T_basis]=generate_M_T_triangle(left,right,bottom,top,h_partition,basis_type);
end 


%Guass quadrature's points and weights on the refenrece triangle and reference interval.
[Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle]=generate_Gauss_reference_triangle(9);
%The following line is necessary if we need to deal with stress or Robin boundary condition. 
%Otherwise, we can comment this line to save time and memory.
[Gauss_coefficient_reference_1D,Gauss_point_reference_1D]=generate_Gauss_reference_1D(4);


%Assemble the matrix.
number_of_elements=2*N1_partition*N2_partition;
number_of_edges=(N1_partition+1)*N2_partition+N1_partition*(2*N2_partition+1);

if basis_type==2||basis_type==1
    matrix_size=[(N1_basis+1)*(N2_basis+1) (N1_basis+1)*(N2_basis+1)];
    vector_size=(N1_basis+1)*(N2_basis+1);
    number_of_FE_nodes=(N1_basis+1)*(N2_basis+1);
elseif basis_type==10
    matrix_size=[number_of_edges number_of_edges];
    vector_size=number_of_edges;
    number_of_FE_nodes=number_of_edges;
end

if basis_type==2
    trial_basis_index=[1 2 3 4 5 6];
    test_basis_index=[1 2 3 4 5 6];
elseif basis_type==1||basis_type==10
    trial_basis_index=[1 2 3];
    test_basis_index=[1 2 3];
end


A1=assemble_matrix_from_volume_integral_triangle_index('function_lamda',M_partition,T_partition,T_basis,T_basis,number_of_elements,matrix_size,trial_basis_index,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,1,0,basis_type,1,0);
A2=assemble_matrix_from_volume_integral_triangle_index('function_mu',M_partition,T_partition,T_basis,T_basis,number_of_elements,matrix_size,trial_basis_index,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,1,0,basis_type,1,0);
A3=assemble_matrix_from_volume_integral_triangle_index('function_mu',M_partition,T_partition,T_basis,T_basis,number_of_elements,matrix_size,trial_basis_index,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,1,basis_type,0,1);
A4=assemble_matrix_from_volume_integral_triangle_index('function_lamda',M_partition,T_partition,T_basis,T_basis,number_of_elements,matrix_size,trial_basis_index,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,1,basis_type,1,0);
A5=assemble_matrix_from_volume_integral_triangle_index('function_mu',M_partition,T_partition,T_basis,T_basis,number_of_elements,matrix_size,trial_basis_index,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,1,0,basis_type,0,1);
A6=assemble_matrix_from_volume_integral_triangle_index('function_lamda',M_partition,T_partition,T_basis,T_basis,number_of_elements,matrix_size,trial_basis_index,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,1,0,basis_type,0,1);
A7=assemble_matrix_from_volume_integral_triangle_index('function_mu',M_partition,T_partition,T_basis,T_basis,number_of_elements,matrix_size,trial_basis_index,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,1,basis_type,1,0);
A8=assemble_matrix_from_volume_integral_triangle_index('function_lamda',M_partition,T_partition,T_basis,T_basis,number_of_elements,matrix_size,trial_basis_index,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,1,basis_type,0,1);


A=[A1+2*A2+A3  A4+A5; A6+A7  A8+2*A3+A2];

%Assemble the vector.
b1=assemble_vector_from_volume_integral_triangle_index('function_f1',M_partition,T_partition,T_basis,number_of_elements,vector_size,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,0);
b2=assemble_vector_from_volume_integral_triangle_index('function_f2',M_partition,T_partition,T_basis,number_of_elements,vector_size,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,0);

b=[b1;b2];


%Get the information matrices for boundary nodes and boundary edges.
if basis_type==2||basis_type==1
    [boundary_nodes,boundary_edges]=generate_boundary_nodes_edges_Stokes(N1_basis,N2_basis,N1_partition,N2_partition);
elseif basis_type==10
    [boundary_nodes,boundary_edges]=generate_boundary_nodes_edges_elasticity_CR(N1_partition,N2_partition);
end


[A,b]=treat_Dirichlet_boundary_Stokes('function_g1','function_g2',A,b,boundary_nodes,M_basis,number_of_FE_nodes);


%Compute the numerical solution
r=A\b;


%Get the finite element solution for u1, u2 and p.
uh1=r(1:number_of_FE_nodes);
uh2=r(number_of_FE_nodes+1:2*number_of_FE_nodes);







