function result=FE_solution_1D(x,uh_local,vertices,basis_type,derivative_degree)
%Xiaoming He, 10/08/2011.
%Evaluate a finite element solution at (x,y) which is in a 1D element T.
%We will use "FE" to replace "finite element" in the comments.
%uh_local: the values of numerical solution at all the nodes of FE basis functions in the triangular element T.
%vertices: the coordinates of all vertices of the 1D element T.
%basis_type: the type of the FE.
%basis_index: the index of basis function to specify which basis function we want to use.
%derivative_degree:the derivative degree of the FE basis function.


result=0;
number_of_local_basis=length(uh_local);
for i=1:number_of_local_basis
    result=result+uh_local(i)*local_basis_1D(x,vertices,basis_type,i,derivative_degree);
end