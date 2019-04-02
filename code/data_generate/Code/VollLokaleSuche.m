function [matxOut, stutz, localSearch1] = VollLokaleSuche(C, B2, am, amz, stutz, sheet, i, matxOut, Margin, PopI, rotind, auflosungsF, localSearch, gewFlache, gewSt, gewKoll, StutzPX)
% Lokale Suche nach besserer Platzierung für Verkippen und Stegbeschädigung
% Einfache brute-force Methode der Lokalen Suche

% Abwägung, wie weit sich Verschiebung lohnt
ref=size(C,2);
if localSearch==1 % unsichere Teile anders behandeln
    maxVersch=ref*gewKoll/gewFlache*0.5; % *0.5
    maxVerschX=maxVersch/size(am,1);
    maxVerschY=maxVersch/size(am,2);
else % Kippende Teile
    maxVersch=ref*gewKoll/gewFlache;
    maxVerschX=maxVersch/size(am,1);
    maxVerschY=maxVersch/size(am,2);
end
% weiter als 62 in x und 17 in y wird auch auf einem endlosen Gitter nicht
% helfen
if maxVerschX > 62
    maxVerschX = 62;
end
if maxVerschY > 17
    maxVerschY =17;
end

% fallen oder stabil anstreben
if size(am,1)>size(am,2)
    lang=size(am,1);
    kurz=size(am,2);
else
    lang=size(am,2);
    kurz=size(am,1);
end

if lang>90 * auflosungsF && kurz>30 * auflosungsF % Teil soll stabil // TODO automatisieren?
    localSearch1=98;
    verbesserung = 0;
    oriBewert=matxOut(6)*gewKoll+matxOut(7)*gewSt; % Bewertung
    oriStutz=zeros(2);
    kandStutz=zeros(2);
    
    for k = 0:1:maxVerschY
        for j = 0:1:maxVerschX
            if k==0 && j==0
                break
            end
            kandStutz(1)=stutz(1)+j;
            kandStutz(2)=stutz(2)+k;
            
            if (kandStutz(1)+size(am,1)-1+Margin <= size(C,1)) && (kandStutz(2)+size(am,2)-1+Margin <= size(C,2)) % muss auf sheet passen
                if (j*size(am,1)+k*size(am,2))<maxVersch % darf zusammen maximale Verschiebung nicht überschreiten
                    
                    [matxOut1, localSearch2]=BewertungLS(C, B2, am, kandStutz, i, Margin, PopI, rotind, auflosungsF);
                    kandBewert=matxOut1(6)*gewKoll+(matxOut1(7)+10*matxOut1(8))*gewSt+(j*size(am,1)+k*size(am,2))/ref*gewFlache;
                    if kandBewert<oriBewert
                        koll = 0;
                        brflag = 0;
                        for n = 1:size(amz,1)% keine Überlappung: nicht nach oben in ein Teil schieben!
                            for m = 1:size(amz,2)
                                try
                                if sheet(kandStutz(1)+n-1, kandStutz(2)+m-1)==1 && amz(n,m)==1 % out of bounds Exception
                                    % disp('Kollision verhindert!')
                                    koll=1;
                                    brflag=1;
                                    break
                                end
                                catch
                                   save('LokaleSucheOOB.mat','kandStutz', 'amz', 'n', 'm')
                                   error('LokaleSuche Out of Bounds        :(')
                                end
                            end
                            if brflag==1
                                break
                            end
                        end
                        
                        if koll==0 % Nur wenn keine Kollision: platzieren!
                            oriBewert=kandBewert;
                            oriStutz(1)= kandStutz(1);
                            oriStutz(2)= kandStutz(2);
                            matxOut = matxOut1;
                            localSearch1=localSearch2;
                            verbesserung =1;
                        end
                    end
                end
            end
        end
    end
    if verbesserung == 1
        stutz(1)=oriStutz(1);
        stutz(2)=oriStutz(2);
        
    end
    
    
else % Teil soll fallen, immer in x-Richtung verschieben
    localSearch1=99;
    % TODO: Drehrichtung prüfen
    if size(am,1)<60
        % Verschiebung prüfen
        verschX = max(StutzPX)-stutz(1);
        if verschX < maxVerschX && stutz(1)+verschX+size(am,1)< size(C,1) % wirklich verschieben, wenn nötige Verschiebung kleiner als maxVersch und noch auf dem Blech 
            stutz(1)=stutz(1)+verschX;
            matxOut(1) = matxOut(1)+verschX; % neuer Schwerpunkt
            matxOut(3) = matxOut(3)+verschX; % neuer Gaspunkt
            
            try
            testFall=1; % testen, ob wirklich kein Steg vorhanden
            for j=1:size(am,1)
                for k=1:size(am,2)
                    if C(j+stutz(1)-1+Margin,k+stutz(2)-1+Margin)==1 && am(j,k)==1
                        testFall=0;
                    end
                end
            end
            catch
               save('VollLSFehlerFallen.mat','j','k','stutz','C', 'am')
               error(':( Ich kann nicht mehr!')
            end
            
            if testFall==1 % Wenn kein Steg fallen
                matxOut(5)=5;
            elseif testFall==0 % wenn Steg verkippen
                matxOut(5)=4;
            end
        end
    end
end

end