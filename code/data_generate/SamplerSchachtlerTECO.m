% SmartNesting, gewrappt f�rs TECO
% All rights reserved to TRUMPF Werkzeugmaschinen GmbH + Co. KG, Germany
% Frederick Struckmeier, released 31.07.2018
% Version 1.0

function outp = SamplerSchachtlerTECO(Teile, anzTeile, zulRot, Margin, auflosungsF, gewFlache, gewKoll, gewSt, Bild, Farbbild, Ident, Seed)

% Parsen, denn Input ist strings

% Position der Umbr�che finden
k = strfind(Teile, '[');
Teile1 = cell(length(k),1);
% Zahlen dazwischen in array aufnehmen
for i = 1:length(k)
    
    if i == length(k)
        substr = extractAfter(Teile, k(i));
    else
        substr = extractBetween(Teile, k(i), k(i+1));
    end
    substr = erase(substr,"[");
    substr = erase(substr,"]");
    substr = char(substr);
    Teile1{i,1} = str2num(substr);
end

%Teile1 ist ein cell array der Form {[Teil], [Teil]...[Teil]}
%Rotation ist ein Array der From [rot, rot...rot]
%zulRot ist ein Array der From [zulRot, zulRot...zulRot]
%dim(Teile1) =  dim(Rotation) = dim(zulRot)

Teile = Teile1;
%Rotation = char(Rotation);
%Rotation = str2num(Rotation);
anzTeile = str2num(char(anzTeile));
zulRot = char(zulRot);
zulRot = str2num(zulRot);
Margin = str2double(Margin);
auflosungsF = str2double(auflosungsF);
gewFlache = str2double(gewFlache);
gewKoll = str2double(gewKoll);
gewSt = str2double(gewSt);
Bild = str2double(Bild);
Farbbild = str2double(Farbbild);

rng(mod(str2num(Seed), 2^32));
numOfPieces = anzTeile;

numCores = feature('numcores');
p = parpool(numCores);



parfor i=1:32
    try
    A = [3000,1500]; % Gr��e der Tafel in mm
    D = cell(1,1); % Array f�r Bilder
    E = cell(1,1); % Array f�r Output-Daten
    C = getAuflage(A, auflosungsF);
    
    Teile_Rand = cell(numOfPieces,1);
   
    %zulRot_Rand = zeros(numOfPieces); 
    %Rotation_Rand = Rotation;
    zulRot_Rand =  zulRot;
    Rotation_Rand = zeros(1,numOfPieces);
    for piece=1:numOfPieces
        index = randi(length(Teile));
        Teile_Rand{piece,1} = Teile{index,1};
        %Rotation_Rand(piece) = Rotation(index)
        %Rotation_Rand(piece) = zulRot(randperm(length(zulRot)));
        
        Rotation_Rand(piece) =  zulRot(randi(length(zulRot)));
    end
    %Rotation_Rand = str2num(Rotation_Rand);
    [B, B2, BLeng] = getPartsTECO(Teile_Rand, auflosungsF, zulRot_Rand); % Liste der Teile
        
    Id = int2str(i); 

    Pop = horzcat(1:1:BLeng, 0, Rotation_Rand);
    [Pop(1, BLeng+1), D{1}, E{1}]= FitnessBLSN(A, B, B2, C, Pop(1, :), zulRot_Rand, auflosungsF, Margin, BLeng, gewFlache, gewKoll, gewSt, Id);
    Pop(1, BLeng+1);

    LosungGrafik(D{1}, E{1}, C, Id);
    
 
    fileID = fopen(strcat(Id,'_configuration.txt'),'w');
    for row = 1:numOfPieces
        fprintf(fileID,'%6.2f',Rotation_Rand(row));
         fprintf(fileID,'%s',' ');
        fprintf(fileID,'%8.2f',Teile_Rand{row,1});
        fprintf(fileID,'%s\r\n',' ');

    end
    %fprintf(fileID,'%6.2f',Rotation_Rand);
    fclose(fileID);
    catch
        warning('FAIL :-(');
        
    end    
        
end
end