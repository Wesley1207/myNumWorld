function r=triangular_local_basis(x,y,vertices,basis_type,basis_index,derivative_degree_x,derivative_degree_y)
%Xiaoming He, 07/01/2009.
%We will use "FE" to replace "finite element" in the comments.
%This is for the local basis functions of triangular FE.
%x,y: the coordinates of the point where we want to evaluate the local FE basis function.
%left_lower_point: The coordinates of the left-lower vertice of the current element whose FE local basis we are evaluating. 
%h_partition: the step size of the partition.
%basis_type: the type of the FE.
%basis_type=1:2D linear FE.
%basis_type=2:2D Lagrange quadratic FE.
%basis_index: the index of basis function to specify which basis function we want to use.
%derivative_degree_x:the derivative degree of the FE basis function with respect to x.
%derivative_degree_y:the derivative degree of the FE basis function with respect to y.

%J is the Jacobi matrix of the affine mapping, which has four elememts J_11, J_12, J_21, J_22.
%J_det is the determinant of J.
%x_hat,y_hat: the '\hat{x}' and '\hat{y}' in the affine mapping, which are for the coordinates of
%the reference element(see my "Notes for tool box of standard triangular FE" section 1-3).

%More explanation is in my "Notes for tool box of standard triangular FE" section 1-4.

J_11=vertices(1,2)-vertices(1,1);
J_12=vertices(1,3)-vertices(1,1);
J_21=vertices(2,2)-vertices(2,1);
J_22=vertices(2,3)-vertices(2,1);
J_det=J_11*J_22-J_12*J_21;

x_hat=(J_22*(x-vertices(1,1))-J_12*(y-vertices(2,1)))/J_det;
y_hat=(-J_21*(x-vertices(1,1))+J_11*(y-vertices(2,1)))/J_det;

if derivative_degree_x==0&&derivative_degree_y==0
    r=triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,0);
elseif derivative_degree_x==1&&derivative_degree_y==0
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,0)*J_22+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,1)*(-J_21))/J_det;
elseif derivative_degree_x==0&&derivative_degree_y==1
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,0)*(-J_12)+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,1)*J_11)/J_det;
elseif derivative_degree_x==2&&derivative_degree_y==0
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,2,0)*J_22^2+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,2)*J_21^2+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,1)*(-2*J_21*J_22))/J_det^2;
elseif derivative_degree_x==0&&derivative_degree_y==2
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,2,0)*J_12^2+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,2)*J_11^2+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,1)*(-2*J_11*J_12))/J_det^2;
elseif derivative_degree_x==1&&derivative_degree_y==1
    r=(triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,2,0)*(-J_22*J_12)+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,0,2)*(-J_21*J_11)+triangular_reference_basis(x_hat,y_hat,basis_type,basis_index,1,1)*(J_21*J_12+J_11*J_22))/J_det^2;
end