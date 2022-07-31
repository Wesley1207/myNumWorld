function r=triangular_reference_basis(x,y,basis_type,basis_index,derivative_degree_x,derivative_degree_y)
%Xiaoming He, 07/01/2009.
%We will use "FE" to replace "finite element" in the comments.
%This is the reference FE basis function on triangle ABC where A=(0,0), B=(1,0) and C=(0,1).
%x,y: the coordinates of the point where we want to evaluate the reference FE basis function.
%basis_type: the type of the FE.
%basis_type=1:2D linear FE.
%basis_type=2:2D Lagrange quadratic FE.
%basis_index: the index of FE basis function to specify which FE basis function we want to use.
%derivative_degree_x:the derivative degree of the FE basis function with respect to x.
%derivative_degree_y:the derivative degree of the FE basis function with respect to y.
%More explanation is in my "Notes for tool box of standard triangular FE" section 1-4.


if basis_type==2
    
    if derivative_degree_x==0&&derivative_degree_y==0
        
        if basis_index==1
            r=1-3*x-3*y+2*x.^2+2*y.^2+4*x.*y;
        elseif basis_index==2
            r=2*x.^2-x;
        elseif basis_index==3
            r=2*y.^2-y;
        elseif basis_index==4
            r=4*x-4*x.^2-4*x.*y;
        elseif basis_index==5
            r=4*x.*y;
        elseif basis_index==6
            r=4*y-4*y.^2-4*x.*y;
        end
             
    elseif derivative_degree_x==1&&derivative_degree_y==0
 
        if basis_index==1
            r=-3+4*x+4*y;
        elseif basis_index==2
            r=4*x-1;
        elseif basis_index==3
            r=0;
        elseif basis_index==4
            r=4-8*x-4*y;
        elseif basis_index==5
            r=4*y;
        elseif basis_index==6
            r=-4*y;
        end           

                      
    elseif derivative_degree_x==0&&derivative_degree_y==1
            
        if basis_index==1
            r=-3+4*y+4*x;
        elseif basis_index==2
            r=0;
        elseif basis_index==3
            r=4*y-1;
        elseif basis_index==4
            r=-4*x;
        elseif basis_index==5
            r=4*x;
        elseif basis_index==6
            r=4-8*y-4*x;
        end
      
    elseif derivative_degree_x==2&&derivative_degree_y==0  
        
        if basis_index==1
            r=4;
        elseif basis_index==2
            r=4;
        elseif basis_index==3
            r=0;
        elseif basis_index==4
            r=-8;
        elseif basis_index==5
            r=0;
        elseif basis_index==6
            r=0;
        end

    elseif derivative_degree_x==0&&derivative_degree_y==2 

        if basis_index==1
            r=4;
        elseif basis_index==2
            r=0;
        elseif basis_index==3
            r=4;
        elseif basis_index==4
            r=0;
        elseif basis_index==5
            r=0;
        elseif basis_index==6
            r=-8;
        end

    elseif derivative_degree_x==1&&derivative_degree_y==1 

        if basis_index==1
            r=4;
        elseif basis_index==2
            r=0;
        elseif basis_index==3
            r=0;
        elseif basis_index==4
            r=-4;
        elseif basis_index==5
            r=4;
        elseif basis_index==6
            r=-4;
        end 
      
    end


elseif basis_type==1

    if derivative_degree_x==0&&derivative_degree_y==0
        
        if basis_index==1
            r=1-x-y;
        elseif basis_index==2
            r=x;
        elseif basis_index==3
            r=y;
        end

    elseif derivative_degree_x==1&&derivative_degree_y==0
        
        if basis_index==1
            r=-1;
        elseif basis_index==2
            r=1;
        elseif basis_index==3
            r=0;
        end

    elseif derivative_degree_x==0&&derivative_degree_y==1
        
        if basis_index==1
            r=-1;
        elseif basis_index==2
            r=0;
        elseif basis_index==3
            r=1;
        end
        
    end
       
end