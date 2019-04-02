% SmartNesting, gewrappt f�rs TECO
% All rights reserved to TRUMPF Werkzeugmaschinen GmbH + Co. KG, Germany
% Frederick Struckmeier, released 31.07.2018
% Version 1.0

function outp = SchachtlerTECO(Teile, Rotation, zulRot, Margin, auflosungsF, gewFlache, gewKoll, gewSt, Bild, Farbbild, Ident)

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

Teile = Teile1;
Rotation = char(Rotation);
Rotation = str2num(Rotation);
zulRot = char(zulRot);
zulRot = str2num(zulRot);
Margin = str2double(Margin);
auflosungsF = str2double(auflosungsF);
gewFlache = str2double(gewFlache);
gewKoll = str2double(gewKoll);
gewSt = str2double(gewSt);
Bild = str2double(Bild);
Farbbild = str2double(Farbbild);

A = [3000,1500]; % Gr��e der Tafel in mm
D = cell(1,1); % Array f�r Bilder
E = cell(1,1); % Array f�r Output-Daten
C = getAuflage(A, auflosungsF);
[B, B2, BLeng] = getPartsTECO(Teile, auflosungsF, zulRot); % Liste der Teile


% Bewertung

Pop = horzcat(1:1:BLeng, 0, Rotation);
[Pop(1, BLeng+1), D{1}, E{1}]= FitnessBLSN(A, B, B2, C, Pop(1, :), zulRot, auflosungsF, Margin, BLeng, gewFlache, gewKoll, gewSt, Ident);
outp = Pop(1, BLeng+1);

if Bild == 1
    Ident = char(Ident);
    spstr = ['Schachtelung ', Ident, '.png'];
    imwrite(D{1},spstr);
end

if Farbbild == 1
    LosungGrafik(D{1}, E{1}, C, Ident);
end

end