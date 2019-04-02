function LosungGrafik(Image, matxOut, C, Ident)
% Bringt double Grafik in RGB Grafik
% Input: codierter Sheet
% Output: Hübsche Grafik der Lösung wird abgespeichert

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Daten aus matxOut verarbeiten
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mCountStab=0;
mCountUns=0;
mCountKipp=0;
mCountFall=0;
mCountMM=0;
mCountEinst=0;

for i=1:size(matxOut,1)
    
    switch matxOut(i,5)
        case 2
            mCountStab=mCountStab+1;
        case 3
            mCountUns=mCountUns+1;
        case 4
            mCountKipp=mCountKipp+1;
        case 5
            mCountFall=mCountFall+1;
    end
    
    mCountMM=mCountMM+matxOut(i,7)*100;
    mCountEinst=mCountEinst+matxOut(i,8);
end

% disp(['Es werden ' num2str(mCountStab) ' Teile stabil liegen bleiben, ' num2str(mCountKipp) ' Teile kippen, ' num2str(mCountFall) ' Teile fallen und ' num2str(mCountUns) ' Teile können nicht eindeutig zugeordnet werden.']);
% disp(['Es gibt ' num2str(mCountEinst) ' Einstiche über Steg und insgesamt werden ' num2str(mCountMM) ' mm über Steg geschnitten.']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sheet vorbearbeiten
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% BWconncomp regions holen

Image1 = zeros(size(Image,1),size(Image,2));
for i=1:size(Image,1)
    for j=1:size(Image,2)
        if Image(i,j)==1 || Image(i,j)==2 || Image(i,j)==3 || Image(i,j)==4 || Image(i,j)==5
            Image1(i,j)=1;
        end
    end
end

cc = bwconncomp(Image1,4);

countStab=0;
countUns=0;
countKipp=0;
countFall=0;
% Wert einlesen und füllen
for i=1:length(cc.PixelIdxList)
    t1=cc.PixelIdxList{i};
    for j=1:length(t1)
        if Image(t1(j))>1 && Image(t1(j))<6
            kodierung=Image(t1(j));
            switch Image(t1(j))
                case 2
                    countStab = countStab+1;
                case 3
                    countUns = countUns+1;
                case 4
                    countKipp = countKipp+1;
                case 5
                    countFall = countFall+1;
            end
            break
        end
    end
    
    % füllen
    for j=1:length(t1)
        if Image(t1(j))==1
            Image(t1(j))=kodierung;
        end
    end
    
end


% Stege schreiben
for i = 1:size(Image,1)
    for j = 1:size(Image,2)
        if C(i,j)==1 % Stegspitze
            if Image(i,j)<6 % Nur Restgitter und Teil überschreiben, keine Schwer- oder Gaspunkte
                switch Image(i,j)
                    case 0 % Restgitter
                        Image(i,j)=7;
                    case 2  % Teil stabil
                        Image(i,j)=10;
                    case 3 % Teil unsicher
                        Image(i,j)=11;
                    case 4 % Teil kippt
                        Image(i,j)=12;
                    case 5 % Teil fällt GIBT ES NICHT
                        Image(i,j)=13; 
                end
            end
        elseif C(i,j)==2 % Steg
            if Image(i,j)<6 % Nur Restgitter und Teil überschreiben, keine Schwer- oder Gaspunkte
                switch Image(i,j)
                    case 0 % Restgitter
                        Image(i,j)=6;
                    case 2  % Teil stabil
                        Image(i,j)=14;
                    case 3 % Teil unsicher
                        Image(i,j)=15;
                    case 4 % Teil kippt
                        Image(i,j)=16;
                    case 5 % Teil fällt
                        Image(i,j)=17;
                end
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RGB erstellen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rgb=zeros(size(Image,1),size(Image,2),3);

for i=1:size(Image,1)
    for j=1:size(Image,2)
        
        switch Image(i,j)
            case 0 % Restgitter: schwarz
                rgb(i,j,1)=0;
                rgb(i,j,2)=0;
                rgb(i,j,3)=0;
                % case 1 ist für Teile in der Kollisionsvermeidung
                % reserviert
            case 2 % stabiles Teil: grün
                rgb(i,j,1)=143;
                rgb(i,j,2)=188;
                rgb(i,j,3)=143;
            case 3 % unsicheres Teil: orange
                rgb(i,j,1)=255;
                rgb(i,j,2)=165;
                rgb(i,j,3)=0;
            case 4 % kippendes Teil: rot
                rgb(i,j,1)=255;
                rgb(i,j,2)=69;
                rgb(i,j,3)=0;
            case 5 % fallendes Teil: blau
                rgb(i,j,1)=50;
                rgb(i,j,2)=20;
                rgb(i,j,3)=250;
            case 6 % Steg: grau
                rgb(i,j,1)=128;
                rgb(i,j,2)=128;
                rgb(i,j,3)=128;
            case 7 % Stegspitze: gelb
                rgb(i,j,1)=255;
                rgb(i,j,2)=255;
                rgb(i,j,3)=0;
            case 8 % Schwerpunkt: grün
                rgb(i,j,1)=0;
                rgb(i,j,2)=255;
                rgb(i,j,3)=0;
            case 9 % Gaspunkt: weiß
                rgb(i,j,1)=255;
                rgb(i,j,2)=255;
                rgb(i,j,3)=255;
            case 10 % Stegspitze stabil
                rgb(i,j,1)=0;
                rgb(i,j,2)=255;
                rgb(i,j,3)=0;
            case 11 % Stegspitze unsicher
                rgb(i,j,1)=255;
                rgb(i,j,2)=100;
                rgb(i,j,3)=0;
            case 12 % Stegspitze kippt
                rgb(i,j,1)=50;
                rgb(i,j,2)=50;
                rgb(i,j,3)=50;
            case 13 % Stegspitze fällt GIBT ES NICHT
                rgb(i,j,1)=100;
                rgb(i,j,2)=100;
                rgb(i,j,3)=255;
            case 14 % Steg stabil
                rgb(i,j,1)=136;
                rgb(i,j,2)=158;
                rgb(i,j,3)=136;
            case 15 % Steg unsicher
                rgb(i,j,1)=192;
                rgb(i,j,2)=147;
                rgb(i,j,3)=64;
            case 16 % Steg kippt
                rgb(i,j,1)=192;
                rgb(i,j,2)=99;
                rgb(i,j,3)=64;
            case 17 % Steg fällt
                rgb(i,j,1)=89;
                rgb(i,j,2)=74;
                rgb(i,j,3)=189;
            otherwise % Da ist was schiefgegangen
                error('Falsche Kodierung des Bildes!')
        end
    end
end

% disp(['Es wurden ' num2str(countStab + countKipp + countFall + countUns) ' Teile im Bild dargestellt.'])
% if countUns==0
%     disp(['Es werden ' num2str(countStab) ' Teile stabil liegen bleiben, ' num2str(countKipp) ' Teile kippen und ' num2str(countFall) ' Teile fallen.']);
% else
%     disp(['Es werden ' num2str(countStab) ' Teile stabil liegen bleiben, ' num2str(countKipp) ' Teile kippen, ' num2str(countFall) ' Teile fallen und ' num2str(countUns) ' Teile können nicht eindeutig zugeordnet werden.']);
% end

rgb=uint8(rgb);
Ident = char(Ident);
imwrite(rgb,['Schachtelung farbe ', Ident, '.png']);

end

