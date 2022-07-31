function [M,T]=generate_M_T_triangle(left,right,bottom,top,h_partition,basis_type)
%Xiaoming He, 07/01/2009.
%Generate the matrix M and T for a triangular mesh of a rectangular domain [left,right]*[bottom,top].
%We will use "FE" to replace "finite element" in the comments.
%h_partition: the step size of the partition.
%We first form a rectangular mesh with element size h_partition(1)*h_partition(2). 
%The triangular mesh is formed by cutting each rectangle into two triangles along the diagonal line from left-top vertex to the right-bottom vertex.
%We will use "FE" to replace "finite element" in the comments.
%basis_type: the type of the FE.
%basis_type=1:2D linear FE.
%basis_type=2:2D Lagrange quadratic FE.
%M stores the coordinates of all nodes for the type of finite element specified by "basis_type".
%M(i,j) is the ith coordinate of the jth node.
%T stores the global indices of the nodes of every element for the type of finite element specified by "basis_type".
%T(i,j) stores the global index of the ith node in th jth element.
%The rules for the index are in my "Notes for tool box of standard triangular FE" section 1-2.

%Note: the grid points of the partition are the same as the nodes of 2D linear FE basis functions.
%Hence the index of the partition is actually the same as that of 2D linear FE, so are M and T.
%However, sometimes the grid points of the partition are different from the nodes of FE basis functions.
%For example,2D Lagrange quadratic FE has 6 nodes correpsonding to the 6 nodal local FE basis functions in each element, 
%but the partition has only 3 grid points for each element,i.e, the 3 vertices.
%So,we may have 2 kinds of M and T, which are M_partition,T_partition and M_basis, T_basis.
%M_partition: store the coordinates of all the grid points of the partition,not FE.
%T_partition: store the global indices of the grid points of every element for the partition,not FE.
%M_basis: store the coordinates of all the nodes for the FE,not the partition.
%T_basis: store the global indices of the nodes of every element for FE,not the partition.
%For 2D linear FE, M_partition,T_partition are the same as M_basis,T_basis.
%For 2D Langrange quadratic FE, they are different. The following 4 rows are what they store for 2D Langrange quadratic FE.
%M_partition only stores the coordinates of all grid points,i.e., vertices of all elements.
%M_basis stores the coordinates of all the nodes of FE.
%T_partition only stores the global indices of the 3 vertices for every element.
%T_basis stores the global indices of the all the 6 nodes of FE for every element.
%The global index for T_partition is different the global index for T_basis:
%the global index for T_partition only counts the vertices of all the element;
%the global index for T_basis counts all the nodes for 2D Lagrange quadratic FE.

%N1 is the number of the sub-intervals of the partition in x-direction.
%N2 is the number of the sub-intervals of the partition in y-direction.
%tnp:total number of all the nodes of FE,including inner nodes and boundary nodes.
%More explanation about M,T,Q,row,column is in my "Standard 2-dimension FE tool boxes notes" sections 2-5 and 2-6.

h=h_partition;

if basis_type==1

   N1=(right-left)/h(1);
   N2=(top-bottom)/h(2);
   tnp=(N1+1)*(N2+1);
   M=zeros(2,tnp);
   T=zeros(3,2*N1*N2);
   Q=zeros(N1+1,N2+1);

   for j=1:tnp
      if mod(j,N2+1)==0
         M(1,j)=left+(j/(N2+1)-1)*h(1);
         M(2,j)=top;
      else
         M(1,j)=left+fix(j/(N2+1))*h(1);
         M(2,j)=bottom+(mod(j,N2+1)-1)*h(2);
      end
   end

   for i=1:N1+1
      for j=1:N2+1
         Q(i,j)=(i-1)*(N2+1)+j;
      end
   end

%Go through all rectangles in the partition. 
%For the nth rectangle, store the information of its two triangular elements whose element indices are 2n-1 and 2n.
   for n=1:N1*N2
   
      if mod(n,N2)==0
         row=N2;
         column=n/N2;
      else
         row=mod(n,N2);
         column=fix(n/N2)+1;
      end
   
      T(1,2*n-1)=Q(column,row);
      T(2,2*n-1)=Q(column+1,row);
      T(3,2*n-1)=Q(column,row+1);  
  
      T(1,2*n)=Q(column,row+1);
      T(2,2*n)=Q(column+1,row);
      T(3,2*n)=Q(column+1,row+1);  
    
   end

elseif basis_type==2

   N1=(right-left)/h(1);
   N2=(top-bottom)/h(2);
   dh=h/2;
   dN1=N1*2;
   dN2=N2*2;
   tnp=(dN1+1)*(dN2+1);
   M=zeros(2,tnp);
   T=zeros(3,2*N1*N2);
   Q=zeros(dN1+1,dN2+1);

   for j=1:tnp
      if mod(j,dN2+1)==0
         M(1,j)=left+(j/(dN2+1)-1)*dh(1);
         M(2,j)=top;
      else
         M(1,j)=left+fix(j/(dN2+1))*dh(1);
         M(2,j)=bottom+(mod(j,dN2+1)-1)*dh(2);
      end
   end

   for i=1:dN1+1
      for j=1:dN2+1
         Q(i,j)=(i-1)*(dN2+1)+j;
      end
   end

%Go through all rectangles in the partition. 
%For the nth rectangle, store the information of its two triangular elements whose element indices are 2n-1 and 2n.
   for n=1:N1*N2
   
      if mod(n,N2)==0
         row=N2;
         column=n/N2;
      else
         row=mod(n,N2);
         column=fix(n/N2)+1;
      end
   
      T(1,2*n-1)=Q(2*column-1,2*row-1);
      T(2,2*n-1)=Q(2*column+1,2*row-1); 
      T(3,2*n-1)=Q(2*column-1,2*row+1);
      T(4,2*n-1)=Q(2*column,2*row-1);
      T(5,2*n-1)=Q(2*column,2*row);
      T(6,2*n-1)=Q(2*column-1,2*row);


      T(1,2*n)=Q(2*column-1,2*row+1);
      T(2,2*n)=Q(2*column+1,2*row-1);
      T(3,2*n)=Q(2*column+1,2*row+1);
      T(4,2*n)=Q(2*column,2*row);
      T(5,2*n)=Q(2*column+1,2*row);
      T(6,2*n)=Q(2*column,2*row+1); 

   end
end
