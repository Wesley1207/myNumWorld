function check_FE_solution_error_1D_linear
%Xiaoming He, 10/08/2011.


format short e

%Check the 2D linear finite element.
basis_type=101

%The problem domain is [left,right]*[bottom,top].
left=0;
right=1;


Gauss_point_number=4;


h_partition=1/4
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);
infinity_error_4=FE_solution_error_infinity_norm_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
L2_error_4=FE_solution_error_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
H1_error_4=FE_solution_error_1D(uh,'exact_solution_derivative',left,right,h_partition,basis_type,1,Gauss_point_number)



h_partition=1/8
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);
infinity_error_8=FE_solution_error_infinity_norm_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
L2_error_8=FE_solution_error_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
H1_error_8=FE_solution_error_1D(uh,'exact_solution_derivative',left,right,h_partition,basis_type,1,Gauss_point_number)

h_partition=1/16
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);
infinity_error_16=FE_solution_error_infinity_norm_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
L2_error_16=FE_solution_error_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
H1_error_16=FE_solution_error_1D(uh,'exact_solution_derivative',left,right,h_partition,basis_type,1,Gauss_point_number)

h_partition=1/32
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);
infinity_error_32=FE_solution_error_infinity_norm_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
L2_error_32=FE_solution_error_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
H1_error_32=FE_solution_error_1D(uh,'exact_solution_derivative',left,right,h_partition,basis_type,1,Gauss_point_number)




h_partition=1/64
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);
infinity_error_64=FE_solution_error_infinity_norm_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
L2_error_64=FE_solution_error_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
H1_error_64=FE_solution_error_1D(uh,'exact_solution_derivative',left,right,h_partition,basis_type,1,Gauss_point_number)




h_partition=1/128
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);
infinity_error_128=FE_solution_error_infinity_norm_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
L2_error_128=FE_solution_error_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
H1_error_128=FE_solution_error_1D(uh,'exact_solution_derivative',left,right,h_partition,basis_type,1,Gauss_point_number)



h_partition=1/256
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);
infinity_error_256=FE_solution_error_infinity_norm_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
L2_error_256=FE_solution_error_1D(uh,'exact_solution',left,right,h_partition,basis_type,0,Gauss_point_number)
H1_error_256=FE_solution_error_1D(uh,'exact_solution_derivative',left,right,h_partition,basis_type,1,Gauss_point_number)


