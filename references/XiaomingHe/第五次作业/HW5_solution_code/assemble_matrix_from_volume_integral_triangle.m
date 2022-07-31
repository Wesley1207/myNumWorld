function r=assemble_matrix_from_volume_integral_triangle(coefficient_function_name,M_partition,T_partition,T_basis_trial,T_basis_test,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,trial_basis_type,trial_derivative_degree_x,trial_derivative_degree_y,test_basis_type,test_derivative_degree_x,test_derivative_degree_y)
%Xiaoming He, 07/01/2009.
%Assemble a matrix from a volume inegral on the whole domain with a triangular mesh.
%We will use "FE" to replace "finite element" in the comments.
%The integrand of the volume integral must be in the following format:
%a coefficient function * a trial FE basis function(or its derivatives) * a test FE basis function (or its derivatives).
%coefficient_function_name: the coefficient function of the integrand.
%M_partition: store the coordinates of all the grid points of the partition,not FE.
%T_partition: store the global indices of the grid points of every element for the partition,not FE.
%T_basis: store the global indices of the nodes of every element for FE,not the partition.
%T_basis_trial: T_basis for the trial basis function.
%T_basis_test: T_basis for the test basis function.
%The explanation for M_partition,T_partition,T_basis is in generate_M_T_triangular.m.
%number_of_trial_local_basis: the number of local FE basis functions for the trial function in a local element.
%number_of_test_local_basis: the number of local FE basis functions for the test function in a local element.
%number_of_element: the number of the local triangluar elements of the partition.
%N1_basis,N2_basis:The N1 and N2 for the basis,not the partition.
%N1 is the number of the sub-intervals of the partition in x-direction.
%N2 is the number of the sub-intervals of the partition in y-direction.
%Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle: the Gauss coefficients and Gauss points on the reference triangular element.
%trial_basis_type:the type of the trial FE basis function.
%trial_basis_type=1:2D linear FE.  
%trial_basis_type=2:2D Lagrange quadratic FE.
%trial_basis_index: the index of trial FE basis function to specify which trial FE basis function we want to use.
%trial_derivative_degree_x:the derivative degree of the trial FE basis function with respect to x.
%trial_derivative_degree_y:the derivative degree of the trial FE basis function with respect to y.
%test_basis_type:the type of the test FE basis function.
%test_basis_type=1:2D linear FE.  
%test_basis_type=2:2D Lagrange quadratic FE.
%test_basis_index: the index of test FE basis function to specify which test FE basis function we want to use.
%test_derivative_degree_x:the derivative degree of the test FE basis function with respect to x.
%test_derivative_degree_y:the derivative degree of the test FE basis function with respect to y.

%vertices: the coordinates of all vertices of a triangular element.
%Gauss_coefficient_local_triangle,Gauss_point_local_triangle: the Gauss coefficients and Gauss points on the local triangular element.

r=sparse(matrix_size(1),matrix_size(2));

%Go through all elements.
%On each element, compute the volume integrals for all possible combinations of trial and test FE basis functions.
%Assemble the values of those volume integrals into the matrix.
for n=1:number_of_elements
   
    vertices=M_partition(:,T_partition(:,n));
    [Gauss_coefficient_local_triangle,Gauss_point_local_triangle]=generate_Gauss_local_triangle(Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,vertices);
   
    for alpha=1:number_of_trial_local_basis
       for beta=1:number_of_test_local_basis      
          temp=Gauss_quadrature_for_volume_integral_trial_test_triangle(coefficient_function_name,Gauss_coefficient_local_triangle,Gauss_point_local_triangle,vertices,trial_basis_type,alpha,trial_derivative_degree_x,trial_derivative_degree_y,test_basis_type,beta,test_derivative_degree_x,test_derivative_degree_y);
          r(T_basis_test(beta,n),T_basis_trial(alpha,n))=r(T_basis_test(beta,n),T_basis_trial(alpha,n))+temp;
       end
    end

end