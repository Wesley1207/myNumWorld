function check_FE_solution_error_1D_quadratic
%Xiaoming He, 10/08/2011.


format short e

%Check the 2D linear finite element.
basis_type=102
%The problem domain is [left,right]*[bottom,top].
left=0;
right=1;


Gauss_point_number=4;


h_partition=1/4
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);


h_partition=1/8
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);

h_partition=1/16
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);

h_partition=1/32
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);


h_partition=1/64
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);


h_partition=1/128
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);

h_partition=1/256
uh=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number);




