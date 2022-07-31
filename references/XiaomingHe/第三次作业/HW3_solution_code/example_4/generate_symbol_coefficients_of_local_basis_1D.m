function generate_symbol_coefficients_of_local_basis_1D

S1=solve('a1*v1^2+b1*v1+c1=1','a1*v2^2+b1*v2+c1=0','a1*(v1+v2)^2/4+b1*(v1+v2)/2+c1=0','a1','b1','c1');
a1=S1.a1
b1=S1.b1
c1=S1.c1

S2=solve('a2*v1^2+b2*v1+c2=0','a2*v2^2+b2*v2+c2=1','a2*(v1+v2)^2/4+b2*(v1+v2)/2+c2=0','a2','b2','c2');
a2=S2.a2
b2=S2.b2
c2=S2.c2

S3=solve('a3*v1^2+b3*v1+c3=0','a3*v2^2+b3*v2+c3=0','a3*(v1+v2)^2/4+b3*(v1+v2)/2+c3=1','a3','b3','c3');
a3=S3.a3
b3=S3.b3
c3=S3.c3