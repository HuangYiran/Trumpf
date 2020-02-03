% 目前存在的问题是：
%   - 出现大量重复系列
%      - 同一个压缩文件中的不存在重复
%      - 不同压缩文件中的重复率几乎达到100%
%   - 评价文件中，存在多个未知空格
%   - rotation种出现了5。本来应该只有0和90的
%   - 各种Teile的数量是否有限制

function outp = get_scores(Teile, anzTeile, zulRot, Margin, auflosungsF, gewFlache, gewKoll, gewSt, Bild, Farbbild, li_teile, li_rot)

% Parsen, denn Input ist strings

% Position der Umbr�che finden
k = strfind(Teile, '[');   % 查找Teile中'['出现的次数，从而确定一共有几种Teile
Teile1 = cell(length(k),1);  % 建立一个[k, 1]矩阵，每行代表一个Teil
% Zahlen dazwischen in array aufnehmen
for i = 1:length(k)   % 提取每个Teil，去掉中括号并把他们转换成number类型后，保持到矩阵中
    
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

Teile = Teile1;   % Teile从str类型变为cell烈性
%Rotation = char(Rotation);
%Rotation = str2num(Rotation);
anzTeile = str2num(char(anzTeile));   % 应该是Teile的数量（注意不是种类的数量）
zulRot = char(zulRot);
zulRot = str2num(zulRot);   % 允许的旋转度数？？？
Margin = str2double(Margin);   % 边界？？？
auflosungsF = str2double(auflosungsF);   % 清晰度？？？
gewFlache = str2double(gewFlache);
gewKoll = str2double(gewKoll);
gewSt = str2double(gewSt);
Bild = str2double(Bild);
Farbbild = str2double(Farbbild);

% randi在为此matlab restart后获得的随机数是一样的，为了避免这种现象，可以在跑randi之前运行
% rng('shuffle)
rng('shuffle');   % ++++++++++++++++++++++++
%numOfPieces = round(anzTeile/2) + randi(anzTeile/2);   % randi(x)生成(0, x)之间的均匀分布的伪随机数
                                                       % 所以numOfPieces是(anzTeile/2, anzTeile)之间的一个数 
                                                       % 可能出现非整除现象，所以randi内的数也要加上round
%tmp = round(anzTeile/2)   % +++++++++++++++++++++
numOfPieces = length(li_teile)   % ++++++++++++++++++++++

numCores = feature('numcores');
%p = parpool(numCores);



%for i=1:2   % 就是paralle + for的意思, 因为并行处理的，所以32个task的顺序不是一致的
try
A = [3000,1500]; % Grosse der Tafel in mm
D = cell(1,1); % Array fuer Bilder
E = cell(1,1); % Array fuer Output-Daten
C = getAuflage(A, auflosungsF);

Teile_Rand = cell(numOfPieces,1);   % 新建cell，用以装选中的Teile
zulRot_Rand =  zulRot;
Rotation_Rand = zeros(1,numOfPieces);   % 生成[1, numOfPieces]全零阵，为何排列数要倒过来？？？
for index=1:numOfPieces
    Teile_Rand{index,1} = Teile{li_teile(index) + 1,1};   % 随机获取一个Teile，扔到Teile_rank中
    Rotation_Rand(index) =  li_rot(index);   % 随机获取一个Rotation，并扔到Rotation_rand中
end
%Rotation_Rand = str2num(Rotation_Rand);
[B, B2, BLeng] = getPartsTECO(Teile_Rand, auflosungsF, zulRot_Rand); % [matrix, schwer points, num_teile]

Id = int2str(0); 

Pop = horzcat(1:1:BLeng, 0, Rotation_Rand);   % horzcat应该是horizon catenate, 应该是合在第一维
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
catch ErrorInfo
    disp(ErrorInfo);
    disp(ErrorInfo.identifier);
    disp(ErrorInfo.message);
    disp(ErrorInfo.stack);
    disp(ErrorInfo.cause);
    %warning('FAIL :-('));
end    
        
%end
end
