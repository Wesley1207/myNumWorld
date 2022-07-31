function [Gauss_coefficient_reference_1D,Gauss_point_reference_1D]=generate_Gauss_reference_1D(Gauss_point_number)
%Xiaoming He, 04/15/2009.
%Generate the Gauss coefficients and Gauss points on the reference interval [-1,1].
%Gauss_point_number:the number of Gauss points in the formula. The Gauss formula depends on it.
%Gauss_coefficient_reference_1D,Gauss_point_reference_1D: the Gauss coefficients and Gauss points on the reference interval.

if Gauss_point_number==4
    Gauss_coefficient_reference_1D=[0.3478548451,0.3478548451,0.6521451549,0.6521451549];
    Gauss_point_reference_1D=[0.8611363116,-0.8611363116,0.3399810436,-0.3399810436];
elseif Gauss_point_number==8
    Gauss_coefficient_reference_1D=[0.1012285363,0.1012285363,0.2223810345,0.2223810345,0.3137066459,0.3137066459,0.3626837834,0.3626837834];
    Gauss_point_reference_1D=[0.9602898565,-0.9602898565,0.7966664774,-0.7966664774,0.5255324099,-0.5255324099,0.1834346425,-0.1834346425];
elseif Gauss_point_number==2
    Gauss_coefficient_reference_1D=[1,1];
    Gauss_point_reference_1D=[-1/sqrt(3),1/sqrt(3)];
end