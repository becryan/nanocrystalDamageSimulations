% finds pairs with index locations and sorts by minimum distance - part of solution from 
%length of array
r=10;
pairs=[];

% example x-y pairs of locations
xys= [-8.5080 ,  21.6870;
   -7.3210,   20.8420;
   -1.5670 ,  29.9890;
    5.9890 ,  13.5250;
    0.3220 ,  15.6690;
    7.2730 ,  11.9140;
   -1.1470 ,  16.5390;
   -5.5760 ,  29.8610;
   -0.4950 ,  30.1250;
   -9.7070 ,  22.9020];

for i=1:r
    for n=i+1:r,
       pairs(end+1,:)=[i n pdist([xys(i);xys(n)])];
    end
end

%sort the pairs by shortest distances

closest_pairs=sortrows(pairs,[3,1]);
       

