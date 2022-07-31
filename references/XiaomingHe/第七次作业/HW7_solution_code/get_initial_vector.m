function r=get_initial_vector(initial_function_name,M_basis)
%Xiaoming He, 07/17/2009.
%Evaluate the initial function at all nodes.
%initial_function_name: the name of the initial funtion.
%M_basis: see the note in "generate_M_T_triangular,m".

number_of_nodes=size(M_basis,2);
r=zeros(number_of_nodes,1);
for i=1:number_of_nodes
    r(i)=feval(initial_function_name,M_basis(1,i),M_basis(2,i));
end