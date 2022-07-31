function [solution_2D,maxerror]=get_2D_solution_and_maximum_error(solution_1D,N1_basis,N2_basis,left,bottom,h_basis)
%Xiaoming He, 07/01/2009.
%Transfer the 1D solution into 2D solution and compute the maximum error at all nodes.
%This function needs to be modified for different boundary conditons.
%Here, this function is for pure Dirichlet boundary condition.
%We will use "FE" to replace "finite element" in the comments.
%N1_basis,N2_basis:The N1 and N2 for the FE basis functions,not the partition.
%N1 is the number of the sub-intervals of the partition in x-direction.
%N2 is the number of the sub-intervals of the partition in y-direction.
%The problem domain is [left,right]*[bottom,top].
%h_basis is the step size for the finite element nodes, not the partition.


maxerror=0;
solution_2D=zeros(N2_basis+1,N1_basis+1);
for i=1:N1_basis+1
   for j=1:N2_basis+1
      solution_2D(j,i)=solution_1D((i-1)*(N2_basis+1)+j);
      temp=solution_2D(j,i)-exact_solution(left+(i-1)*h_basis(1),bottom+(j-1)*h_basis(2));
      if abs(maxerror)<abs(temp)
         maxerror=temp;
      end
   end
end