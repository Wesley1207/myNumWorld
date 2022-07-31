function [maxerror,l2error]=get_maximum_l2_error_1D(solution,N_basis,left,h_basis)
%Xiaoming He, 01/19/2013.
%compute the maximum error at all nodes for 1D problem.

maxerror=0;
l2error=0;
for i=1:N_basis+1
    temp=solution(i)-exact_solution(left+(i-1)*h_basis);
    if abs(maxerror)<abs(temp)
        maxerror=temp;
    end
    l2error=l2error+temp^2;
end
l2error=sqrt(l2error);