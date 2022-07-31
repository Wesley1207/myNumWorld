function check_heat_FE_solution_error_triangle_quadratic
%Xiaoming He, 07/17/2009.
%theta: decide the time discretization algorithm.
%0<=theta<=1.
%theta=0: forward Euler.
%theta=1: backward Euler.
%theta=0.5: Crank-Nicolson.

format short e

%Check the 2D linear finite element.
basis_type=2

%The problem domain is [left,right]*[bottom,top].
left=0;
right=1;
bottom=0;
top=0.75;

Gauss_point_number=9;

initial_t=0;
end_t=1;



%Crank-Nicolson.
theta=0.5

h_partition=[1/4,1/4]
N_t=8
dt=(end_t-initial_t)/N_t;
uh=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%r=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%uh=r(:,N_t+1);
infinity_error_4=FE_solution_error_infinity_norm_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_4=FE_solution_error_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_time_triangle(uh,'exact_solution_x_derivative',end_t,left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_time_triangle(uh,'exact_solution_y_derivative',end_t,left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_4=sqrt(H1_error_x^2+H1_error_y^2)



h_partition=[1/8,1/8]
N_t=32
dt=(end_t-initial_t)/N_t;
uh=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%r=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%uh=r(:,N_t+1);
infinity_error_8=FE_solution_error_infinity_norm_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_8=FE_solution_error_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_time_triangle(uh,'exact_solution_x_derivative',end_t,left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_time_triangle(uh,'exact_solution_y_derivative',end_t,left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_8=sqrt(H1_error_x^2+H1_error_y^2)



h_partition=[1/16,1/16]
N_t=64
dt=(end_t-initial_t)/N_t;
uh=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%r=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%uh=r(:,N_t+1);
infinity_error_16=FE_solution_error_infinity_norm_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_16=FE_solution_error_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_time_triangle(uh,'exact_solution_x_derivative',end_t,left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_time_triangle(uh,'exact_solution_y_derivative',end_t,left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_16=sqrt(H1_error_x^2+H1_error_y^2)



h_partition=[1/32,1/32]
N_t=32
dt=(end_t-initial_t)/N_t;
uh=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%r=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%uh=r(:,N_t+1);
infinity_error_32=FE_solution_error_infinity_norm_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_32=FE_solution_error_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_time_triangle(uh,'exact_solution_x_derivative',end_t,left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_time_triangle(uh,'exact_solution_y_derivative',end_t,left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_32=sqrt(H1_error_x^2+H1_error_y^2)



h_partition=[1/64,1/64]
N_t=64*8
dt=(end_t-initial_t)/N_t;
uh=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%r=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%uh=r(:,N_t+1);
infinity_error_64=FE_solution_error_infinity_norm_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_64=FE_solution_error_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_time_triangle(uh,'exact_solution_x_derivative',end_t,left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_time_triangle(uh,'exact_solution_y_derivative',end_t,left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_64=sqrt(H1_error_x^2+H1_error_y^2)










%Backward Euler.
theta=1

h_partition=[1/4,1/4]
N_t=4^3
dt=(end_t-initial_t)/N_t;
uh=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%r=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%uh=r(:,N_t+1);
infinity_error_4=FE_solution_error_infinity_norm_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_4=FE_solution_error_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_time_triangle(uh,'exact_solution_x_derivative',end_t,left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_time_triangle(uh,'exact_solution_y_derivative',end_t,left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_4=sqrt(H1_error_x^2+H1_error_y^2)



h_partition=[1/8,1/8]
N_t=8^3
dt=(end_t-initial_t)/N_t;
uh=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%r=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%uh=r(:,N_t+1);
infinity_error_8=FE_solution_error_infinity_norm_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_8=FE_solution_error_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_time_triangle(uh,'exact_solution_x_derivative',end_t,left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_time_triangle(uh,'exact_solution_y_derivative',end_t,left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_8=sqrt(H1_error_x^2+H1_error_y^2)



h_partition=[1/16,1/16]
N_t=16^3
dt=(end_t-initial_t)/N_t;
uh=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%r=heat_solver_triangle(left,right,bottom,top,h_partition,basis_type,dt,initial_t,end_t,theta);
%uh=r(:,N_t+1);
infinity_error_16=FE_solution_error_infinity_norm_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
L2_error_16=FE_solution_error_time_triangle(uh,'exact_solution',end_t,left,right,bottom,top,h_partition,basis_type,0,0,Gauss_point_number)
H1_error_x=FE_solution_error_time_triangle(uh,'exact_solution_x_derivative',end_t,left,right,bottom,top,h_partition,basis_type,1,0,Gauss_point_number);
H1_error_y=FE_solution_error_time_triangle(uh,'exact_solution_y_derivative',end_t,left,right,bottom,top,h_partition,basis_type,0,1,Gauss_point_number);
H1_error_16=sqrt(H1_error_x^2+H1_error_y^2)








