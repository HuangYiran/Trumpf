function [outp] = getAuflage(A,auflosungsF)
% Position der Auflagepunkte einlesen
% Zur Zeit alle 33,5/67 mm, Steg 2mm breit (Realit�t 3), 15mm (14,8) zwischen den Spitzen, Spitze 1mm breit
% Abh�ngig von Gr��e der Fl�che

outp = zeros(A(1)*auflosungsF,A(2)*auflosungsF);
xstep = 67 * auflosungsF; % muss beides automatisch bestimmt werden k�nnen
ystep = 15 * auflosungsF;

for i=1:xstep:size(outp,1)
   outp(i,:)=2;
   outp(i+1,:)=2; % Steg ist 2
end

for i=1:xstep:size(outp,1)
    for j=1:ystep:size(outp,2)
        outp(i,j)=1;
        outp(i+1,j)=1; % Stegspitze ist 1
    end
end

end