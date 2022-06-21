function y= getRandomChoice (Proba)

% this function gives you a random choice from 1 to 10 based of MPF  Proba

[P, I] = sort(Proba,'descend');
cdf=zeros(length(P),1);

cdf(1)=P(1);
for i=2:length(P)
    cdf(i)=cdf(i-1)+P(i);
end

x= rand(1,1);
if  (x <= cdf(1))
    y= I(1);
elseif (x > cdf(1)) && (x <= cdf(2))
    y= I(2);
elseif (x > cdf(2)) && (x <= cdf(3))
    y= I(3);
elseif (x > cdf(3)) && (x <= cdf(4))
    y= I(4);
elseif (x > cdf(4)) && (x <= cdf(5))
    y= I(5);
elseif (x > cdf(5)) && (x <= cdf(6))
    y= I(6);
elseif (x > cdf(6)) && (x <= cdf(7))
    y= I(7);
elseif (x > cdf(7)) && (x <= cdf(8))
    y= I(8);
elseif (x > cdf(8)) && (x <= cdf(9))   
    y= I(9);
elseif (x > cdf(9)) && (x <= cdf(10))
    y= I(10);
end

end
