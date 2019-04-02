function [outp1, outp2, BLeng] = getPartsTECO(Teile, auflosungsF, zulRot)

BLeng = size(Teile,1);

outp = Teile; 

% Polygon in Matrix wandeln
outp1 = cell(BLeng,size(zulRot,2));
for i=1:BLeng
    a=outp{i,1}.*auflosungsF;
    minX=min(a(:,1)); % Polygon in ersten Quadranten verschieben
    minY=min(a(:,2));
    for j=1:size(a,1)
        b=a(j,1);
        b=b+abs(minX);
        a(j,1)=b;
    end
    for j=1:size(a,1)
        b=a(j,2);
        b=b+abs(minY);
        a(j,2)=b;
    end
    
    maxX=ceil(max(a(:,1)));
    maxY=ceil(max(a(:,2)));
    am=zeros(maxX, maxY);
    for j=1:maxX
        for k=1:maxY
            am(j,k)=inpolygon(j,k,a(:,1),a(:,2));
        end
    end
    outp1{i,1}=am;
end

% Drehrichtungen berechnen
for i=1:BLeng
    for j=2:size(zulRot,2)
        outp1{i,j}=transpose(outp1{i,1}); % Nur 0 und 90 Grad Drehungen!
    end
end

% Schwerpunkt berechnen

outp2 = zeros(BLeng,4*size(zulRot,2));
for i = 1:BLeng
    [pMx,pMy] = Schwerpunkt(outp{i,1}); % errechnen
    outp2(i,1)=pMx;
    outp2(i,2)=pMy;
    helM=zeros(size(outp1{i,1},1),size(outp1{i,1},2));
    helM(ceil(pMx),ceil(pMy))=1;
    helM(ceil(pMx)+1,ceil(pMy))=1;
    for j = 2:size(zulRot,2)
        helMa=transpose(helM);
        [rotx,roty]=ind2sub([size(helMa,1),size(helMa,2)],find(helMa==1));
        if size(rotx,1)>1
            rotx=rotx(1);
        end
        if size(roty,1)>1
            roty=roty(1);
        end
        outp2(i,1+(j-1)*4)=rotx;
        outp2(i,2+(j-1)*4)=roty;
    end
end

% Gaspunkt berechnen

for i=1:BLeng
    for j=1:size(zulRot,2)
        tam=outp1{i,j};
        for k=1:size(tam,1)
            if tam(k,2)==1 % hängt vom Einstichspunkt ab. Annahme treffen: y-Wert 2 (1 könnte leer sein), linkster x-Wert
                outp2(i,3+(j-1)*4) = k;
                outp2(i,4+(j-1)*4) = 2;
                break
            end
        end
    end
end

outp2 = round(outp2);

end