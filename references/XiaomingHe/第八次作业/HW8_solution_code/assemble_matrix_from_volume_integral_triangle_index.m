function result=assemble_matrix_from_volume_integral_triangle_index(coefficient_function_name,M_partition,T_partition,T_basis_trial,T_basis_test,number_of_elements,matrix_size,trial_basis_index,test_basis_index,Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,trial_basis_type,trial_derivative_degree_x,trial_derivative_degree_y,test_basis_type,test_derivative_degree_x,test_derivative_degree_y)
%Xiaoming He, 09/24/2011.
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
%number_of_element: the number of the local triangluar elements of the partition.
%trial_basis_index: a vector which stores the indices of the local trial basis functions we need for the matrix.
%test_basis_index: a vector which stores the indices of the local test basis functions we need for the matrix.
%Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle: the Gauss coefficients and Gauss points on the reference triangular element.
%trial_basis_type:the type of the trial FE basis function.
%basis_type=0:2D constant FE.
%basis_type=1:2D Lagrange linear FE.
%basis_type=2:2D Lagrange quadratic FE.
%basis_type=10:2D Crouzeix-Raviart FE.
%trial_basis_index: the index of trial FE basis function to specify which trial FE basis function we want to use.
%trial_derivative_degree_x:the derivative degree of the trial FE basis function with respect to x.
%trial_derivative_degree_y:the derivative degree of the trial FE basis function with respect to y.
%test_basis_type:the type of the test FE basis function.
%basis_type=0:2D constant FE.
%basis_type=1:2D Lagrange linear FE.
%basis_type=2:2D Lagrange quadratic FE.
%basis_type=10:2D Crouzeix-Raviart FE.
%test_basis_index: the index of test FE basis function to specify which test FE basis function we want to use.
%test_derivative_degree_x:the derivative degree of the test FE basis function with respect to x.
%test_derivative_degree_y:the derivative degree of the test FE basis function with respect to y.

%vertices: the coordinates of all vertices of a triangular element.
%Gauss_coefficient_local_triangle,Gauss_point_local_triangle: the Gauss coefficients and Gauss points on the local triangular element.

result=sparse(matrix_size(1),matrix_size(2));
trial_basis_index_length=length(trial_basis_index);
test_basis_index_length=length(test_basis_index);

%Go through all elements.
%On each element, compute the volume integrals for all possible combinations of trial and test FE basis functions.
%Assemble the values of those volume integrals into the matrix.
for n=1:number_of_elements
   
    vertices=M_partition(:,T_partition(:,n));
    [Gauss_coefficient_local_triangle,Gauss_point_local_triangle]=generate_Gauss_local_triangle(Gauss_coefficient_reference_triangle,Gauss_point_reference_triangle,vertices);
   
    for j=1:trial_basis_index_length
        alpha=trial_basis_index(j);
        for i=1:test_basis_index_length   
            beta=test_basis_index(i);   
            temp=Gauss_quadrature_for_volume_integral_trial_test_triangle(coefficient_function_name,Gauss_coefficient_local_triangle,Gauss_point_local_triangle,vertices,trial_basis_type,alpha,trial_derivative_degree_x,trial_derivative_degree_y,test_basis_type,beta,test_derivative_degree_x,test_derivative_degree_y);
            result(T_basis_test(beta,n),T_basis_trial(alpha,n))=result(T_basis_test(beta,n),T_basis_trial(alpha,n))+temp;
       end
    end

end