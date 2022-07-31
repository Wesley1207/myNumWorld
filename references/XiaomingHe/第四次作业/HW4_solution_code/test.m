clear;
clc;
left=-1;
right=1;
bottom=-1;
top=1;
Nx=10;
Ny=5;
hx=(right-left)/Nx;
hy=(top-bottom)/Ny;
bt=2;  % basis type
if bt == 1
    Nbx = Nx+1;
    Nby = Ny+1;
elseif bt==2
    Nbx = Nx*2+1;
    Nby = Ny*2+1;
end
[bcNode,bcEdge]=generate_boundary_nodes_edges(Nbx-1, Nby-1,Nx,Ny);
[matP,matT]=generate_M_T_triangle(left,right,bottom,top,[hx,hy],1);
[matPb,matTb]=generate_M_T_triangle(left,right,bottom,top,[hx,hy],bt);

