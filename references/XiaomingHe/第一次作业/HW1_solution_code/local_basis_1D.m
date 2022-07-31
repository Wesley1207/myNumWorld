function result=local_basis_1D(x,vertices,basis_type,basis_index,derivative_degree)
%Xiaoming He, 10/08/2011.
%We will use "FE" to replace "finite element" in the comments.
%This is for the local basis functions of 1D FE.
%x: the coordinate of the point where we want to evaluate the local FE basis function.
%basis_type: the type of the FE.
%basis_type=101:1D linear FE.
%basis_index: the index of basis function to specify which basis function we want to use.
%derivative_degree:the derivative degree of the FE basis function.



if basis_type==101
    
    if derivative_degree==0
        
        if basis_index==1
            result=(vertices(2)-x)/(vertices(2)-vertices(1));
        elseif basis_index==2
            result=(x-vertices(1))/(vertices(2)-vertices(1));
        end

    elseif derivative_degree==1
        
        if basis_index==1
            result=1/(vertices(1)-vertices(2));
        elseif basis_index==2
            result=1/(vertices(2)-vertices(1));
        end
        
    end  

    
end


