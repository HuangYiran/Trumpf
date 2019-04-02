function [matxOut, localSearch] = BewertungLS(C, B2, am, stutz, i, Margin, PopI, rotind, auflosungsF)
% SmartNesting Bewertung der Platzierung eines Teils
% Stabilität für jedes Teil, Kollisionswk.

localSearch=0;
matxOut = [0 0 0 0 0 0 0 0];

% Für Log nötig
Fm=0;
Fp=0;
dm1=0;
dp1=0;
dm2=0;
dp2=0;
Mk1=0;
Mk2=0;

% Stützstellen bestimmen
auflP=zeros(size(am,1),size(am,2));
try
    for j=1:size(am,1)
        for k=1:size(am,2)
            if C(j+stutz(1)-1+Margin,k+stutz(2)-1+Margin)==1 && am(j,k)==1
                auflP(j,k)=1;
            end
        end
    end
catch
    save('errorVar2.mat','stutz', 'j', 'k', 'Margin')
    error(':(   Ich kann nicht weiterrechenen!')
end

% feststellen, ob nur ein Steg oder eine Spitzenreihe
uAufl = sum(max(auflP));
StutzP=find(auflP);
[StutzPX, StutzPY]= ind2sub(size(am,1),StutzP);
StutzPX = StutzPX +stutz(1)+Margin;
StutzPY = StutzPY +stutz(2)+Margin;

% Schwerpunkt und Gaspunkt abhängig von Rotation
matxOut(1) = B2(PopI(i),(1+(rotind-1)*4))+stutz(1)+Margin;
matxOut(2) = B2(PopI(i),(2+(rotind-1)*4))+stutz(2)+Margin;
matxOut(3) = B2(PopI(i),(3+(rotind-1)*4))+stutz(1)+Margin;
matxOut(4) = B2(PopI(i),(4+(rotind-1)*4))+stutz(2)+Margin;

% keine Stützstelle = durchfallen und ein Steg = verkippen abfangen
if uAufl==0 % Abfangen, dass Teile keine Spitzen, aber auf Stegen sein können und nicht fallen, obwohl keine Auflagepunkte.
    FalKipp=0;
    for j=1:size(am,1)
        for k=1:size(am,2)
            if C(j+stutz(1)-1+Margin,k+stutz(2)-1+Margin)==2 && am(j,k)==1
                FalKipp=1;
            end
        end
    end
    if FalKipp==0
        matxOut(5)=5; % Codierung
        % disp('Fallen 1')
    else
        matxOut(6) = 1;
        matxOut(5)=4; % Codierung
        localSearch=2;
        % disp('Verkippen 1')
    end
elseif size(unique(StutzPY),1)==1 % Auflagepunkte in einer Reihe: Teil kippt
    % ACHTUNG: Wenn Stegspitzen mehrere Pixel in einer Dimension, dann
    % hier abfangen!
    matxOut(6) = 1;
    matxOut(5)=4; % Codierung
    localSearch=3;
    % disp('Verkippen 1')
elseif size(unique(StutzPX),1)==2 || size(unique(StutzPX),1)==1 % Auflagepunkte in einer Reihe: Teil kippt
    % ACHTUNG: Wenn Stegspitzen mehrere Pixel in einer Dimension, dann
    % hier abfangen!
    matxOut(6) = 1;
    matxOut(5)=4; % Codierung
    localSearch=4;
    % disp('Verkippen 1')
else
    
    % Konvexe Hülle der Stützstellen
    try
        StutzP = convhull(StutzPX, StutzPY);
    catch
        disp('Konvexe Hülle konnte nicht berechnet werden')
        save('FehlerKonvH.mat', 'StutzPX', 'StutzPY', 'uAufl', 'auflP', 'amz', 'sheet');
        error('Konvexe Hülle')
    end
    StutzPX = StutzPX(StutzP);
    StutzPY = StutzPY(StutzP);
    
    % Prüfen, ob Kippberechnung nötig,
    if inpolygon(matxOut(3),matxOut(4),StutzPX,StutzPY) && inpolygon(matxOut(1),matxOut(2),StutzPX, StutzPY)
        % Leere Anweisung, da Kippberechnung nicht nötig, beide Kraftpunkte innerhalb der Stützpunkte: Teil stabil
        matxOut(5)=2; % Codierung
        % disp('Stabil 1')
    else
        
        % Schneidgaskraft und Schwerkraft
        % Winkel auf Teil bestimmen
        eWinkel = pi/2; % Finde Bauteilwinkel am Einstichpunkt /// TODO automatisch Winkel finden
        Fp = (0.002/2)^2 * 1500000 * eWinkel; % Durchmesser der Schneiddüse: 2mm ; Schneidgasdruck: 15 bar = 100000 Pa
        tFlache = (length(find(am)))/auflosungsF^2; % Fläche des Bauteils in mm^2
        Fm = 9.81*0.000015712*tFlache; % F=m*g, Annahme 2mm Baustahl mit Dichte 7856 kg/m^3, ein cm^2 wiegt 0,000015712kg

        % Prüfen, ob Schwerpunkt außerhalb der Stützpunkte.
        [in,on] = inpolygon(matxOut(1),matxOut(2),StutzPX, StutzPY);
        if in==0 || on==1
            matxOut(6) = 1; % Schwerpunkt außerhalb der Stützpunkte. Teil kippt.
            matxOut(5)=4; % Codierung
            % disp('Verkippen 2')
        else
            
            % Eigentlich richtig: nächsten Auflagepunkt finden, Achsen
            % durch diesen Auflagepunkt bestimmen, daraus Drehmoment an
            % beiden Achsen
            %
            % nächsten Auflagepunkt aus konvexer Hülle finden
            shortIndex=0;
            shortDist=10000; % TODO dynamisch groß genug wählen
            for j=1:length(StutzPX)
                dist = sqrt( (matxOut(3)-StutzPX(j))^2 + (matxOut(4)-StutzPY(j))^2 );
                if dist<shortDist
                    shortIndex =j;
                    shortDist = dist;
                end
            end
            
            % Achsen finden
            if shortIndex==1
                ch1=[StutzPX(shortIndex),StutzPY(shortIndex),0]; % nähster Punkt der Konvexen Hülle
                ch2=[StutzPX(length(StutzPX)-1),StutzPY(length(StutzPX)-1),0]; % Punkt davor. Erster und letzter Punkt derselbe!
                ch3=[StutzPX(shortIndex+1),StutzPY(shortIndex+1),0]; % Punkt danach
            elseif shortIndex==length(StutzPX)
                ch1=[StutzPX(shortIndex),StutzPY(shortIndex),0]; % nähster Punkt der Konvexen Hülle
                ch2=[StutzPX(shortIndex-1),StutzPY(shortIndex-1),0]; % Punkt davor
                ch3=[StutzPX(2),StutzPY(2),0]; % Punkt danach. Erster und letzter Punkt derselbe!
            else
                ch1=[StutzPX(shortIndex),StutzPY(shortIndex),0]; % nähster Punkt der Konvexen Hülle
                ch2=[StutzPX(shortIndex-1),StutzPY(shortIndex-1),0]; % Punkt davor
                ch3=[StutzPX(shortIndex+1),StutzPY(shortIndex+1),0]; % Punkt danach
            end
            
            % Prüfen, ob selbe Stützstelle vorkommt
            SelberXWert12 = ch1(1)+1 == ch2(1) || ch1(1)-1 == ch2(1);
            SelberXWert13 = ch1(1)+1 == ch3(1) || ch1(1)-1 == ch3(1);
            SelberYWert12 = ch1(2)+1 == ch2(2) || ch1(2)-1 == ch2(2);
            SelberYWert13 = ch1(2)+1 == ch3(2) || ch1(2)-1 == ch3(2);
            
            % Wenn ja, dann verschieben
            if SelberXWert12 || SelberYWert12 % ch1 und ch2 benachbart
                if shortIndex > 2
                    ch2=[StutzPX(shortIndex-2),StutzPY(shortIndex-2),0];
                elseif shortIndex == 2
                    ch2=[StutzPX(length(StutzPX)),StutzPY(length(StutzPX)),0];
                else
                    ch2=[StutzPX(length(StutzPX)-1),StutzPY(length(StutzPX)-1),0];
                end
            end
            if SelberXWert13 || SelberYWert13 % ch1 und ch3 benachbart
                if shortIndex == length(StutzPX)-1
                    ch3=[StutzPX(2),StutzPY(2),0];
                elseif shortIndex == length(StutzPX)
                    ch3=[StutzPX(3),StutzPY(3),0];
                else
                    ch3=[StutzPX(shortIndex+2),StutzPY(shortIndex+2),0];
                end
            end
            
            pM=[matxOut(1),matxOut(2),0];
            pP=[matxOut(3),matxOut(4),0];
            
            % Abstände zu Achsen
            dm1=norm(cross(ch1-ch2,pM-ch2))/norm(ch1-ch2);
            dp1=norm(cross(ch1-ch2,pP-ch2))/norm(ch1-ch2)*(-1); % Gaskraft wird negativ angenommen
            dm2=norm(cross(ch1-ch3,pM-ch3))/norm(ch1-ch3);
            dp2=norm(cross(ch1-ch3,pP-ch3))/norm(ch1-ch3)*(-1); % Gaskraft wird negativ angenommen
            
%             % Loggen
%             fileID2 = fopen('Hebel Log.txt','a');
%             fprintf(fileID2,'%7.3f ', i);
%             fprintf(fileID2,'%7.3f ', Fm); 
%             fprintf(fileID2,'%7.3f ', Fp); 
%             fprintf(fileID2,'%7.3f ', dm1); 
%             fprintf(fileID2,'%7.3f ', dp1); 
%             fprintf(fileID2,'%7.3f ', dm2); 
%             fprintf(fileID2,'%7.3f ',dp2);
            
            % Drehmomente berechnen
            % TOODO: dafür Kräfte zerlegen? zur Zeit wird das Drehmoment zu
            % hoch eingeschätzt: Volle Kraft auf beiden Achsen
            % Mk1 = dm1*Fm*(dm1/(dm1-dp1))+dp1*Fp*(-dp1/(dm1-dp1));
            % Mk2 = dm2*Fm*(dm2/(dm2-dp2))+dp2*Fp*(-dp2/(dm2-dp2));
            
            Mk1 = dm1*Fm+dp1*Fp;
            Mk2 = dm2*Fm+dp2*Fp;
            
%             fprintf(fileID2,'%7.3f ',Mk1);
%             fprintf(fileID2,'%7.3f \r\n',Mk2);
%             fclose(fileID2);
            
            % Drehmoment an relevanter Kippachse. Richtung von Schwerkraft positiv,
            % von Gaskraft negativ

            if Mk1 < -10 || Mk2 < -10 % Teil kippt. in Nmm
                matxOut(6) = 1;
                matxOut(5)=4; % Codierung
                localSearch=5;
                % disp('Verkippen 3')
            elseif (-10 < Mk1 && Mk1 < 10) || (-10 < Mk2 && Mk2 < 10) % Teil unsicher 
                matxOut(6) = 0.5;
                matxOut(5)=3; % Codierung
                localSearch=6;
%                 fileID3 = fopen('HebelLogUnsicher.txt','a');
%                 fprintf(fileID3,'%7.3f ', i);
%                 fprintf(fileID3,'%7.3f ', Fm);
%                 fprintf(fileID3,'%7.3f ', Fp);
%                 fprintf(fileID3,'%7.3f ', dm1);
%                 fprintf(fileID3,'%7.3f ', dp1);
%                 fprintf(fileID3,'%7.3f ', dm2);
%                 fprintf(fileID3,'%7.3f ',dp2);
%                 fprintf(fileID3,'%7.3f ',Mk1);
%                 fprintf(fileID3,'%7.3f \r\n',Mk2);
%                 for i =1:length(StutzPX)
%                     fprintf(fileID3,'%7.3f ', StutzPX(i));
%                     fprintf(fileID3,'%7.3f \r\n', StutzPY(i));
%                 end
%                 fclose(fileID3);
                % disp('Unsicher')
            else % Teil sicher
                matxOut(5)=2; % Codierung
                % disp('Stabil 2')
            end
            
        end
    end
end

% if matxOut(5)==4
%     fileID4 = fopen('HebelLogKipp.txt','a');
%     fprintf(fileID4,'%7.3f ', i);
%     fprintf(fileID4,'%7.3f ', Fm);
%     fprintf(fileID4,'%7.3f ', Fp);
%     fprintf(fileID4,'%7.3f ', dm1);
%     fprintf(fileID4,'%7.3f ', dp1);
%     fprintf(fileID4,'%7.3f ', dm2);
%     fprintf(fileID4,'%7.3f ',dp2);
%     fprintf(fileID4,'%7.3f ',Mk1);
%     fprintf(fileID4,'%7.3f \r\n',Mk2);
%     
%     for i =1:length(StutzPX)
%         fprintf(fileID4,'%7.3f ', StutzPX(i));
%         fprintf(fileID4,'%7.3f \r\n', StutzPY(i));
%     end
%     fclose(fileID4);
% end

% if matxOut(5)==2
%     fileID5 = fopen('HebelLogSicher.txt','a');
%     fprintf(fileID5,'%7.3f ', i);
%     fprintf(fileID5,'%7.3f ', Fm);
%     fprintf(fileID5,'%7.3f ', Fp);
%     fprintf(fileID5,'%7.3f ', dm1);
%     fprintf(fileID5,'%7.3f ', dp1);
%     fprintf(fileID5,'%7.3f ', dm2);
%     fprintf(fileID5,'%7.3f ',dp2);
%     fprintf(fileID5,'%7.3f ',Mk1);
%     fprintf(fileID5,'%7.3f \r\n',Mk2);
%     
%     for i =1:length(StutzPX)
%         fprintf(fileID5,'%7.3f ', StutzPX(i));
%         fprintf(fileID5,'%7.3f \r\n', StutzPY(i));
%     end
%     fclose(fileID5);
% end

% Stege

brflag=0;
% Einstich über Steg: bSt+1
for j = -1:1:1
    for k = -1:1:1
        if C(matxOut(3)+j,matxOut(4)+k)>0 % pP~ hat Margin und stutz schon aufaddiert
            matxOut(8) = 1;
            % disp('Einstich über Steg!')
            brflag=1;
            break
        end
    end
    if brflag==1
        break
    end
end

% Schnittlänge über Steg: bSt*Schnittlänge in mm/2
schnittl=0;
bound = bwboundaries(am);
try
    bound = bound{1};
catch
    save('errorBound.mat','am', 'bound', 'i')
    error('Boundary Berechnung Error!        :(')
end

try
    for j=1:size(bound,1)
        for k=1:size(bound,2)
            if C(bound(j,1)+stutz(1)-1+Margin,bound(k,2)+stutz(2)-1+Margin)>0
                schnittl = schnittl+1;
            end
        end
    end
    matxOut(7)=matxOut(7)+schnittl/100;
    
catch
    a = j+stutz(1)-1+Margin;
    b = k+stutz(2)-1+Margin;
    save('SchnittError.mat','C','a','b', 'bound', 'am', 'i')
    error(':(        Es ist leider etwas schiefgegangen...')
end

% Logge verkippberechnung für jedes Teil
% hchar = ['VerkippLog' num2str(i) '.mat'];
% save(hchar,'pMx','pMy', 'pPx', 'pPy', 'StutzPX', 'StutzPY', 'stutz')

end