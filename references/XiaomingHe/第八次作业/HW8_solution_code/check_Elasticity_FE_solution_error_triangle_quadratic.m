function check_Elasticity_FE_solution_error_triangle_quadratic
%Xiaoming He, 10/02/2011.
%Check the Taylor-Hood finite element solution error for a Stokes equation.


format short e

%The problem domain is [left,right]*[bottom,top].
left=0;
right=1;
bottom=0;
top=1;

basis_type=2;

Gauss_point_number=9;


h_partition=[1/8,1/8]
[uh1,uh2]=Elasticity_solver_triangle(left,right,bottom,top,h_partition,basis_type);
basis_index=[1 2 3 4 5 6];

uh1_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_infinity_error_8=max(uh1_infinity_error,uh2_infinity_error)

uh1_L2_error=FE_solution_error_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_L2_error=FE_solution_error_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_L2_error_8=sqrt(uh1_L2_error^2+uh2_L2_error^2)

uh1_H1_error_x=FE_solution_error_triangle_index(uh1,'u1_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh1_H1_error_y=FE_solution_error_triangle_index(uh1,'u1_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh1_H1_error=sqrt(uh1_H1_error_x^2+uh1_H1_error_y^2);
uh2_H1_error_x=FE_solution_error_triangle_index(uh2,'u2_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh2_H1_error_y=FE_solution_error_triangle_index(uh2,'u2_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh2_H1_error=sqrt(uh2_H1_error_x^2+uh2_H1_error_y^2);
uh_H1_error_8=sqrt(uh1_H1_error^2+uh2_H1_error^2)




h_partition=[1/16,1/16]
[uh1,uh2]=Elasticity_solver_triangle(left,right,bottom,top,h_partition,basis_type);
basis_index=[1 2 3 4 5 6];

uh1_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_infinity_error_16=max(uh1_infinity_error,uh2_infinity_error)

uh1_L2_error=FE_solution_error_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_L2_error=FE_solution_error_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_L2_error_16=sqrt(uh1_L2_error^2+uh2_L2_error^2)

uh1_H1_error_x=FE_solution_error_triangle_index(uh1,'u1_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh1_H1_error_y=FE_solution_error_triangle_index(uh1,'u1_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh1_H1_error=sqrt(uh1_H1_error_x^2+uh1_H1_error_y^2);
uh2_H1_error_x=FE_solution_error_triangle_index(uh2,'u2_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh2_H1_error_y=FE_solution_error_triangle_index(uh2,'u2_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh2_H1_error=sqrt(uh2_H1_error_x^2+uh2_H1_error_y^2);
uh_H1_error_16=sqrt(uh1_H1_error^2+uh2_H1_error^2)




h_partition=[1/32,1/32]
[uh1,uh2]=Elasticity_solver_triangle(left,right,bottom,top,h_partition,basis_type);
basis_index=[1 2 3 4 5 6];

uh1_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_infinity_error_32=max(uh1_infinity_error,uh2_infinity_error)

uh1_L2_error=FE_solution_error_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_L2_error=FE_solution_error_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_L2_error_32=sqrt(uh1_L2_error^2+uh2_L2_error^2)

uh1_H1_error_x=FE_solution_error_triangle_index(uh1,'u1_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh1_H1_error_y=FE_solution_error_triangle_index(uh1,'u1_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh1_H1_error=sqrt(uh1_H1_error_x^2+uh1_H1_error_y^2);
uh2_H1_error_x=FE_solution_error_triangle_index(uh2,'u2_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh2_H1_error_y=FE_solution_error_triangle_index(uh2,'u2_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh2_H1_error=sqrt(uh2_H1_error_x^2+uh2_H1_error_y^2);
uh_H1_error_32=sqrt(uh1_H1_error^2+uh2_H1_error^2)





h_partition=[1/64,1/64]
[uh1,uh2]=Elasticity_solver_triangle(left,right,bottom,top,h_partition,basis_type);
basis_index=[1 2 3 4 5 6];

uh1_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_infinity_error_64=max(uh1_infinity_error,uh2_infinity_error)

uh1_L2_error=FE_solution_error_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_L2_error=FE_solution_error_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_L2_error_64=sqrt(uh1_L2_error^2+uh2_L2_error^2)

uh1_H1_error_x=FE_solution_error_triangle_index(uh1,'u1_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh1_H1_error_y=FE_solution_error_triangle_index(uh1,'u1_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh1_H1_error=sqrt(uh1_H1_error_x^2+uh1_H1_error_y^2);
uh2_H1_error_x=FE_solution_error_triangle_index(uh2,'u2_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh2_H1_error_y=FE_solution_error_triangle_index(uh2,'u2_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh2_H1_error=sqrt(uh2_H1_error_x^2+uh2_H1_error_y^2);
uh_H1_error_64=sqrt(uh1_H1_error^2+uh2_H1_error^2)





h_partition=[1/128,1/128]
[uh1,uh2]=Elasticity_solver_triangle(left,right,bottom,top,h_partition,basis_type);
basis_index=[1 2 3 4 5 6];

uh1_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_infinity_error_128=max(uh1_infinity_error,uh2_infinity_error)

uh1_L2_error=FE_solution_error_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_L2_error=FE_solution_error_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_L2_error_128=sqrt(uh1_L2_error^2+uh2_L2_error^2)

uh1_H1_error_x=FE_solution_error_triangle_index(uh1,'u1_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh1_H1_error_y=FE_solution_error_triangle_index(uh1,'u1_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh1_H1_error=sqrt(uh1_H1_error_x^2+uh1_H1_error_y^2);
uh2_H1_error_x=FE_solution_error_triangle_index(uh2,'u2_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh2_H1_error_y=FE_solution_error_triangle_index(uh2,'u2_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh2_H1_error=sqrt(uh2_H1_error_x^2+uh2_H1_error_y^2);
uh_H1_error_128=sqrt(uh1_H1_error^2+uh2_H1_error^2)






h_partition=[1/256,1/256]
[uh1,uh2]=Elasticity_solver_triangle(left,right,bottom,top,h_partition,basis_type);
basis_index=[1 2 3 4 5 6];

uh1_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_infinity_error=FE_solution_error_infinity_norm_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_infinity_error_256=max(uh1_infinity_error,uh2_infinity_error)

uh1_L2_error=FE_solution_error_triangle_index(uh1,'u1_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh2_L2_error=FE_solution_error_triangle_index(uh2,'u2_exact_solution',left,right,bottom,top,h_partition,basis_index,basis_type,0,0,Gauss_point_number);
uh_L2_error_256=sqrt(uh1_L2_error^2+uh2_L2_error^2)

uh1_H1_error_x=FE_solution_error_triangle_index(uh1,'u1_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh1_H1_error_y=FE_solution_error_triangle_index(uh1,'u1_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh1_H1_error=sqrt(uh1_H1_error_x^2+uh1_H1_error_y^2);
uh2_H1_error_x=FE_solution_error_triangle_index(uh2,'u2_exact_solution_x_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,1,0,Gauss_point_number);
uh2_H1_error_y=FE_solution_error_triangle_index(uh2,'u2_exact_solution_y_derivative',left,right,bottom,top,h_partition,basis_index,basis_type,0,1,Gauss_point_number);
uh2_H1_error=sqrt(uh2_H1_error_x^2+uh2_H1_error_y^2);
uh_H1_error_256=sqrt(uh1_H1_error^2+uh2_H1_error^2)
