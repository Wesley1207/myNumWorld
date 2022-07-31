function result=FE_solution_triangle_index(x,y,uh_local,vertices,basis_index,basis_type,derivative_degree_x,derivative_degree_y)
%Xiaoming He, 09/04/2011.
%Evaluate a finite element solution at (x,y) which is in a triangular element T.
%We will use "FE" to replace "finite element" in the comments.
%uh_local: the values of numerical solution at all the nodes of FE basis functions in the triangular element T.
%vertices: the coordinates of all vertices of the triangular element T.
%basis_type: the type of the FE.
%basis_type=0:2D constant FE.
%basis_type=1:2D Lagrange linear FE.
%basis_type=2:2D Lagrange quadratic FE.
%basis_type=10:2D Crouzeix-Raviart FE.
%basis_index: the index of basis function to specify which basis function we want to use.
%derivative_degree_x:the derivative degree of the FE basis function with respect to x.
%derivative_degree_y:the derivative degree of the FE basis function with respect to y.

result=0;
basis_index_length=length(basis_index);

for k=1:basis_index_length   
    i=basis_index(k);
    result=result+uh_local(i)*triangular_local_basis(x,y,vertices,basis_type,i,derivative_degree_x,derivative_degree_y);
end