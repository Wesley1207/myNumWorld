function check_FE_solution_error_triangle_linear
%Xiaoming He, 07/04/2009.


format short e

%Check the 2D linear finite element.
basis_type=1

%The problem domain is [left,right]*[bottom,top].
left=-1;
right=1;
bottom=-1;
top=1;

Gauss_point_number=9;


h_partition=[1/4,1/4]
uh=poisson_solver_triangle(left,right,bottom,top,h_partition,basis_type);
infinity_error_4=FE_solution_error_infinity_norm_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_4=FE_solution_error_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_triangle(uh,'exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_triangle(uh,'exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_4=sqrt(H1_error_x^2+H1_error_y^2)



h_partition=[1/8,1/8]
uh=poisson_solver_triangle(left,right,bottom,top,h_partition,basis_type);
infinity_error_8=FE_solution_error_infinity_norm_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_8=FE_solution_error_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_triangle(uh,'exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_triangle(uh,'exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_8=sqrt(H1_error_x^2+H1_error_y^2)


h_partition=[1/16,1/16]
uh=poisson_solver_triangle(left,right,bottom,top,h_partition,basis_type);
infinity_error_16=FE_solution_error_infinity_norm_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_16=FE_solution_error_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_triangle(uh,'exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_triangle(uh,'exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_16=sqrt(H1_error_x^2+H1_error_y^2)


h_partition=[1/32,1/32]
uh=poisson_solver_triangle(left,right,bottom,top,h_partition,basis_type);
infinity_error_32=FE_solution_error_infinity_norm_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_32=FE_solution_error_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_triangle(uh,'exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_triangle(uh,'exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_32=sqrt(H1_error_x^2+H1_error_y^2)


h_partition=[1/64,1/64]
uh=poisson_solver_triangle(left,right,bottom,top,h_partition,basis_type);
infinity_error_64=FE_solution_error_infinity_norm_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_64=FE_solution_error_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_triangle(uh,'exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_triangle(uh,'exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_64=sqrt(H1_error_x^2+H1_error_y^2)


h_partition=[1/128,1/128]
uh=poisson_solver_triangle(left,right,bottom,top,h_partition,basis_type);
infinity_error_128=FE_solution_error_infinity_norm_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_128=FE_solution_error_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_triangle(uh,'exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_triangle(uh,'exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_128=sqrt(H1_error_x^2+H1_error_y^2)


h_partition=[1/256,1/256]
uh=poisson_solver_triangle(left,right,bottom,top,h_partition,basis_type);
infinity_error_256=FE_solution_error_infinity_norm_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_256=FE_solution_error_triangle(uh,'exact_solution',left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_triangle(uh,'exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_triangle(uh,'exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_256=sqrt(H1_error_x^2+H1_error_y^2)

