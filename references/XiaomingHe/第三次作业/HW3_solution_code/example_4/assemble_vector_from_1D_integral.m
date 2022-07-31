function result=assemble_vector_from_1D_integral(coefficient_function_name,M_partition,T_partition,T_basis_test,number_of_test_local_basis,number_of_elements,vector_size,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,test_basis_type,test_derivative_degree)
%Xiaoming He, 10/08/2011.
%Assemble a vector from a 1D inegral on the whole domain.
%We will use "FE" to replace "finite element" in the comments.
%The integrand of the 1D integral must be in the following format:
%a coefficient function * a test FE basis function (or its derivatives).
%coefficient_function_name: the coefficient function of the integrand.
%M_partition: store the coordinates of all the grid points of the partition,not FE.
%T_partition: store the global indices of the grid points of every element for the partition,not FE.
%T_basis: store the global indices of the nodes of every element for FE,not the partition.
%T_basis_test: T_basis for the test basis function.
%The explanation for M_partition,T_partition,T_basis is in generate_M_T_1D.m
%h_partition: the step size of the partition
%number_of_test_local_basis: the number of local FE basis functions for the test function in a local element.
%number_of_element: the number of the local 1D elements of the partition.
%Gauss_coefficient_reference_1D,Gauss_point_reference_1D:the Gauss coefficients and Gauss points on the reference interval [-1,1].
%test_vertices: the coordinates of all vertices of the triangular element for test functions.
%test_basis_type:the type of the test FE basis function.
%test_basis_index: the index of test FE basis function to specify which test FE basis function we want to use.
%test_derivative_degree:the derivative degree of the test FE basis function.

%vertices: the coordinates of the two vertices of a 1D element.
%Gauss_coefficient_local_1D,Gauss_point_local_1D: the Gauss coefficients and Gauss points on the local interval.


result=zeros(vector_size,1);


%Go through all elements.
%On each element, compute the 1D integrals for all test FE basis functions.
%Assemble the values of those 1D integrals into the vector.
for n=1:number_of_elements

    vertices=M_partition(:,T_partition(:,n));
    lower_bound=min(vertices(1),vertices(2));
    upper_bound=max(vertices(1),vertices(2));
    [Gauss_coefficient_local_1D,Gauss_point_local_1D]=generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound);

    for beta=1:number_of_test_local_basis     
        temp=Gauss_quadrature_for_1D_integral_test(coefficient_function_name,Gauss_coefficient_local_1D,Gauss_point_local_1D,vertices,test_basis_type,beta,test_derivative_degree);
        result(T_basis_test(beta,n),1)=result(T_basis_test(beta,n),1)+temp;
    end

end