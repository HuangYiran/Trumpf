function [outp1,outp2] = Schwerpunkt(B1)
% Berechnet den Schwerpunkt eines Polygons
X=B1(:,1);
Y=B1(:,2);
% P=polyshape(X,Y);
% [outp1,outp2] = centroid(P);

% Summen berechnen
xs=0;
for i=1:(size(X,1))
    if i < size(X,1)
        xs = xs+(X(i)+X(i+1))*(X(i)*Y(i+1)-X(i+1)*Y(i));
    elseif i == size(X,1)
        xs = xs + (X(i)+X(1))*(X(i)*Y(1)-X(1)*Y(i));
    end
end
ys=0;
for i=1:(size(Y,1))
    if i < size(X,1)
        ys = ys+(Y(i)+Y(i+1))*(X(i)*Y(i+1)-X(i+1)*Y(i));
    elseif i == size(X,1)
        ys = ys+(Y(i)+Y(1))*(X(i)*Y(1)-X(1)*Y(i));
    end
end

if xs<0
    xs = -xs;
end
if ys<0
    ys = -ys;
end

% A berechnen
A = polyarea(X,Y);

% Outputs berechnen
outp1 = 1/(6*A)*xs;
outp2 = 1/(6*A)*ys;

end
