function y = Accuracy(Alpha,Imgs,Labels)
Num = size(Imgs,2);
y = sum(sign(Alpha'*Imgs)-Labels'==0 )/Num*100;  
end