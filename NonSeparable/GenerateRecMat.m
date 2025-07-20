function [RecMat,Recmat_norm2] = GenerateRecMat(paraFPPA)
%% generate the reconstruction matrix A
WaveName = paraFPPA.WaveName;
RecLev = paraFPPA.RecLev; 
N = RecLev(end);
RecMat =  sparse(N,N);                                                              
for i=1:1:N
    a=zeros(N,1); a(i)=1;
    RecMat(:,i) = waverec(a,RecLev,WaveName);
end
Recmat_norm2 = power_iteration(RecMat);
save RecMat RecMat Recmat_norm2
end