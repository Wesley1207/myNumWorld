function [Gauss_coefficient_local_1D,Gauss_point_local_1D]=generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound)
%Xiaoming He, 04/15/2009.
%Generate the Gauss coefficients and Gauss points on an arbitrary interval [lower_bound,upper_bound] by using affine tranformation.
%Gauss_coefficient_local_1D,Gauss_point_local_1D:the Gauss coefficients and Gauss points on an arbitrary interval.
%Gauss_coefficient_reference,Gauss_point_reference: the Gauss coefficients and Gauss points on the reference interval [-1,1].

Gauss_coefficient_local_1D=(upper_bound-lower_bound)*Gauss_coefficient_reference_1D/2;
Gauss_point_local_1D=(upper_bound-lower_bound)*Gauss_point_reference_1D/2+(upper_bound+lower_bound)/2;