function r=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta)
%Xiaoming He, 07/23/2009.
%Finite element solver for  equation on a triangular mesh.
%The coefficient function K on the left side of the heat equation is a matrix function K=[k11(x,y)  k12(x,y); k21(x,y)  k22(x,y)].
%We will use "FE" to replace "finite element" in the comments.
%The trial FE functions and test FE functions need to be the same.
%So far the code can only handle 2D linear FE and 2D Lagrange quadratic FE.
%For other types of FE, we need to add the corresponding information to code.
%The problem domain is [left,right]*[bottom,top] and the time domain is [initial_t,end_t].
%h_partition is the step size of the space partition.
%basis_type:the type of the FE basis function.
%basis_type=1:2D linear FE.  
%basis_type=2:2D Lagrange quadratic FE.
%dt is the step size of the time partition.
%theta: decide the time discretization algorithm.
%0<=theta<=1.
%theta=0: forward Euler.
%theta=1: backward Euler.
%theta=0.5: Crank-Nicolson.

%N1_basis,N2_basis:The N1 and N2 for the FE basis functions,not the partition.
%N1_partition,N2_partition:The N1 and N2 for the partition,not the FE basis functions.
%N1 is the number of the sub-intervals of the partition in x-direction.
%N2 is the number of the sub-intervals of the partition in y-direction.
%M_partition,T_partition, M_basis,T_basis: see the note in "generate_M_T_triangular,m".
%function_k11,function_k12,function_k21,function_k22: the components of the coefficient matrix function K on the left side of the heat equation.
%function_c: the coefficient function of "u" on the left side of the heat equation.
%funciton_f: the right hand side function of the heat equation.
%function_g: the Dirichelet boundary function in my notes "Notes for tool box of standard triangular FE" section 3-1-1.
%function_q_tilde: the name of the Neumann boundary function \tilde{q}(x,y) in my notes "Notes for tool box of standard triangular FE" section 3-2-2.
%function_q: the name of the Neumann boundary function q(x,y) when p(x,y) is nonzero in my notes "Notes for tool box of standard triangular FE" section 3-1-6.
%function_p: the name of the Robin coefficient function p(x,y) in my notes "Notes for tool box of standard triangular FE" section 3-1-6.
%In this package we assume function_k11,function_k12,function_k21,function_k22,function_c,function_p are independent of time.

%h_basis is the step size for the finite element nodes, not the partition.

N_t=(end_t-initial_t)/dt;

N1_partition=(right-left)/h_partition(1);
N2_partition=(top-bottom)/h_partition(2);

if basis_type==2
    N1_basis=N1_partition*2;
    N2_basis=N2_partition*2;
elseif basis_type==1
    N1_basis=N1_partition;
    N2_basis=N2_partition;
end

if basis_type==2
    h_basis=h_partition/2;
elseif basis_type==1
    h_basis=h_partition;
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
number_of_unknowns=(N1_basis+1)*(N2_basis+1);
matrix_size=[number_of_unknowns number_of_unknowns];
if basis_type==2
    number_of_trial_local_basis=6;
    number_of_test_local_basis=6;
elseif basis_type==1
    number_of_trial_local_basis=3;
    number_of_test_local_basis=3;
end
A1=assemble_matrix_from_volume_integral_triangle('function_k11',M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,1,0,basis_type,1,0);
A2=assemble_matrix_from_volume_integral_triangle('function_k22',M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,1,basis_type,0,1);
A3=assemble_matrix_from_volume_integral_triangle('function_k12',M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,1,basis_type,1,0);
A4=assemble_matrix_from_volume_integral_triangle('function_k21',M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,1,0,basis_type,0,1);

A=A1+A2+A3+A4;
clear A1 A2 A3 A4

%In many cases, there is no term "c*u" in the heat equation, i.e, c=0.
%Hence we can comment the following three lines for those cases.
A5=assemble_matrix_from_volume_integral_triangle('function_c',M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,0,basis_type,0,0);
A=A+A5;
clear A5

%Assemble the mass matrix.
M=assemble_matrix_from_volume_integral_triangle('function_one',M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,0,basis_type,0,0);

%Get the information matrices for boundary nodes and boundary edges.
[boundary_nodes,boundary_edges]=generate_boundary_nodes_edges(N1_basis,N2_basis,N1_partition,N2_partition);

%Deal with Robin boundary condition for the matrix. 
%If we don't have Robin boundary condition at all, we can comment the following line to save time and memory.
A=treat_Robin_boundary_for_matrix_triangle('function_p',A,boundary_edges,M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,basis_type,0,0,basis_type,0,0);

%Get the fixed matrix for the time iteration.
%Part (a) in my "Notes for tool box of standard triangular FE" section 3-1-5, 3-2-2, 3-2-3, 3-3.
if theta~=0
%Not forward Euler.
    A_fixed=M/(theta*dt)+A;
    clear A
else
%forward Euler.  
    A_fixed=M/dt;
    clear M
end


%Initialize the iteration in time.
X_old=get_initial_vector('function_initial',M_basis);
%r=zeros(number_of_unknowns,N_t+1);
%r(:,1)=X_old;

%Iteration in time.
for n=1:N_t
    
    current_time=initial_t+dt*n;
    
%Assemble the load vector.
%Part (b) in my "Notes for tool box of standard triangular FE" section 3-1-5, 3-2-2, 3-2-3,3-3.
    vector_size=number_of_unknowns;
    b1=assemble_vector_from_volume_integral_time_triangle('function_f',current_time,M_partition,T_partition,T_basis,number_of_test_local_basis,number_of_elements,vector_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,0);
    b2=assemble_vector_from_volume_integral_time_triangle('function_f',current_time-dt,M_partition,T_partition,T_basis,number_of_test_local_basis,number_of_elements,vector_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,basis_type,0,0);
    b=theta*b1+(1-theta)*b2;
    
%Part (c) in my "Notes for tool box of standard triangular FE" section 3-1-5, 3-2-2, 3-2-3,3-3.
    if theta~=0
%Not forward Euler.
        b=b+M*X_old/(theta*dt);
    else
%forward Euler.  
        b=b+(A_fixed-A)*X_old;
    end
    
%Part (e) in my "Notes for tool box of standard triangular FE" section 3-2-2, 3-3.    
%Deal with Neumann boundary condition. If we don't have Neumann boundary condition at all, we can comment the following line to save time and memory.
    b=treat_Neumann_boundary_time_triangle('function_q_tilde',current_time,current_time-dt,theta,b,boundary_edges,M_partition,T_partition,T_basis,number_of_test_local_basis,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,basis_type,0,0);

%Part (d) in my "Notes for tool box of standard triangular FE" section 3-2-3, 3-3. 
%Deal with Robin boundary condition. If we don't have Robin boundary condition at all, we can comment the following line to save time and memory.
    b=treat_Robin_boundary_for_vector_time_triangle('function_q',current_time,current_time-dt,theta,b,boundary_edges,M_partition,T_partition,T_basis,number_of_test_local_basis,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,basis_type,0,0);

    
%Deal with Dirichlet boundary condition depending on time.
    if theta~=0
%Not forward Euler. 
        [A_fixed,b]=treat_Dirichlet_boundary_time_theta_triangle('function_g',current_time,dt,theta,A_fixed,b,boundary_nodes,M_basis);
    else
%forward Euler.  
        [A_fixed,b]=treat_Dirichlet_boundary_time_triangle('function_g',current_time,A_fixed,b,boundary_nodes,M_basis);
    end 
    
%Compute the numerical solution.
    X=A_fixed\b;
    
%Update the numerical solution.
    if theta~=0
%Not forward Euler.
        X=(X-X_old)/theta+X_old;
    end

%Prepare for next loop.
    X_old=X;
    %r(:,n+1)=X;
    r=X;
    
    
%Transfer the 1D solution into 2D solution and compute the maximum error at all nodes at the end time.
    if n==N_t
        [solution_2D,maxerror]=get_2D_solution_and_maximum_error_time(end_t,X,N1_basis,N2_basis,left,bottom,h_basis);
        maximum_error_at_all_nodes_of_FE_end_time=maxerror
    end

    
end
