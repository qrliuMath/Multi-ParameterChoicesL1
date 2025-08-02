function RecMat = GenerateRecMat(paraFPPA)
%% generate the reconstruction matrix A
WaveName = paraFPPA.WaveName;
RecLev = paraFPPA.RecLev; 
N = RecLev(end);
RecMat =  sparse(N,N);                                                              
for i=1:1:N
    a = zeros(N,1); a(i)=1;
    RecMat(:,i) = waverec(a,RecLev,WaveName);
end
save RecMat RecMat 
end