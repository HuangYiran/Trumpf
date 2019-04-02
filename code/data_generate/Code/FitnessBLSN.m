function [outp,sheet,matxOutKpl] = FitnessBLSN(A, B, B2, C, PopI, zulRot, auflosungsF, Margin, BLeng, gewFlache, gewKoll, gewSt, Ident)
% Bewertung der Fitness eines Chromosoms mit Bottom Left
% Inputs: Größe der Tafel: 1*2 Matrix; Liste der Teile: m*n Cell Array; Chromosom mit Fitness und Rotation: 1*n Matrix
% Output: Fitness des Chromosoms: Zahl
% Tafel Pixel ist Quadrat mit 1/auflosungsF mm Kantenlänge

sheet=zeros(A(1)*auflosungsF,A(2)*auflosungsF);
sheetX = 2; % Initialisierung Einfügeort in x
outp = 0;
bSt = 0;
erwKoll = 0;
matxOutKpl=zeros(BLeng,8);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Schachtelung rechnen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Für jedes Polygon
for i = 1:BLeng
    % Index der zul Rotation holen
    rotind = find((zulRot==PopI(i+1+BLeng))==1);
    am=B{PopI(i),rotind}; % FEHLER zu viele Argumente zurückgegben. left handside []
    if Margin==0
        amz = am;
    else
        amz=zeros(size(am,1)+2*Margin, size(am,2)+2*Margin);
        for j=1:size(am,1)
            for k=1:size(am,2)
                amz(j+Margin,k+Margin)=am(j,k);
            end
        end
        se=strel('disk',Margin,8);
        amz=imdilate(amz,se);
    end
    
    maxX=size(amz,1);
    maxY=size(amz,2);
    
    % Erst Left dann bottom ausführen
    stutz(1) = sheetX; % Stützvektor für Platzierung
    stutz(2) = size(sheet,2)-maxY+1-Margin;
    
    % Prüfen, ob platzieren überhaupt möglich
    try
        kol=0;
        if maxX+sheetX>size(sheet,1)
            kol=1;
        else
            for j=1:maxX
                for k=1:maxY
                    if sheet(j+stutz(1)-1,k+stutz(2)-1)==1 && amz(j,k)==1
                        kol=1;
                        break
                    end
                end
            end
        end
    catch
        save('errorVar.mat','maxX', 'maxY','stutz','j','k', 'sheet', 'amz', 'sheetX')
        error(':(   Leider ist etwas schiefgegangen. Ich kann nicht weiterrechenen!')
    end
    
    % Abfangen, wenn kein Platzieren möglich
    if kol==1
        outp = outp + 100000;
        continue
    else
        % Prüfen, ob verschieben möglich
        flag=true;
        while(flag)
            leftP=1;
            bottomP=1;
            
            for j=1:maxX % links verschieben
                for k=1:maxY
                    if sheet(max(stutz(1)+j-2,1), stutz(2)+k-1)==1 && amz(j,k)==1
                        leftP=0;
                    end
                end
            end
            
            if leftP==1 % unten nicht prüfen, wenn links möglich
                % Leere Anweisung,um ganzen Block zu überspringen
            else
                for j=1:maxX % unten verschieben
                    for k=1:maxY
                        if sheet(stutz(1)+j-1, max(stutz(2)+k-2,1))==1 && amz(j,k)==1
                            bottomP=0;
                        end
                    end
                end
            end
            
            if leftP==1 && stutz(1)>1 % erst versuchen nach links zu verschieben
                stutz(1) = stutz(1)-1;
            elseif bottomP==1 && stutz(2)>1 % wenn links nicht geht, nach unten
                stutz(2) = stutz(2)-1;
            else % Wenn nicht mehr verschoben werden kann: platzieren
                flag=false;                
            end
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Platzierung bewerten
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    localSearch = 0;
    [matxOut, localSearch, StutzPX] = Bewertung(C, B2, am, stutz, i, Margin, PopI, rotind, auflosungsF); % Hier wirklich nur am nehmen? nicht amz?
    
    % Loggen
    fstr = strcat(Ident, ".txt");
    fileID2 = fopen(fstr,'a');
%     if i==1
%         fprintf(fileID2,'%s \r\n', ["Durchgang ID: ", Ident]);
%     end
    fprintf(fileID2,'%s ', datestr(now));
    fprintf(fileID2,'%6.2f ', i);
%     fprintf(fileID2,'%6.2f ', stutz(1));
%     fprintf(fileID2,'%6.2f ', stutz(2));
%     fprintf(fileID2,'%6.2f ', localSearch);
%     fprintf(fileID2,'%6.2f ', matxOut(5));
%     fprintf(fileID2,'%6.2f ', matxOut(6));
%     fprintf(fileID2,'%6.2f ', matxOut(7));
%     fprintf(fileID2,'%6.2f ', matxOut(8));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Lokale Suche nach Verbesserungen
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if localSearch>0
        [matxOut, stutz, ~] = VollLokaleSuche(C, B2, am, amz, stutz, sheet, i, matxOut, Margin, PopI, rotind, auflosungsF, localSearch, gewFlache, gewSt, gewKoll, StutzPX);
    end
    
    fprintf(fileID2,'%6.2f ', stutz(1));
    fprintf(fileID2,'%6.2f ', stutz(2));
    % fprintf(fileID2,'%6.2f ', localSearch);
    fprintf(fileID2,'%6.2f\r\n', matxOut(5));
    % fprintf(fileID2,'%6.2f ', matxOut(6));
    % fprintf(fileID2,'%6.2f ', matxOut(7));
    % fprintf(fileID2,'%6.2f ', matxOut(8));
    % fprintf(fileID2, '%s\r\n', datestr(now));
    fclose(fileID2);
    
    matxOutKpl(i,:)=matxOut;
    sheet(matxOut(1),matxOut(2))=8; % Schwerpunkt
    sheet(matxOut(3),matxOut(4))=9; % Gaspunkt
    sheet(matxOut(3)+1,matxOut(4)) = matxOut(5); % Codierung
    erwKoll = erwKoll + matxOut(6);
    bSt = bSt + matxOut(7)+ 10*matxOut(8);
    
    % Platzieren
    for j=1:size(am,1)
        for k=1:size(am,2)
            if sheet(j+stutz(1)-1+Margin,k+stutz(2)-1+Margin)==0 && am(j,k)==1 % Gas und Schwerpunkte nicht überschreiben
                sheet(j+stutz(1)-1+Margin,k+stutz(2)-1+Margin)=1;
            end
            
        end
    end
    
    % x-Wert für dynamischen Stützvektor
    for j = size(sheet,1):-1:1
        if sum(sheet(j,:)) > 0
            sheetX=min(j+1,size(sheet,1)); % größer als 1, kleiner als das Sheet
            break
        end
    end
    
end % Platzieren der Teile fertig


% Länge des Blechs
for i = 1:size(sheet,1)
    if sum(sheet(i,:)) > 0
        flache = i/auflosungsF;
    end
end

outp = outp+flache*gewFlache+erwKoll*gewKoll+bSt*gewSt; % Gewichtungsfaktoren als Input
% zusätzlich Butzen berücksichtigen
end