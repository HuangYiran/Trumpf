function [plgs] = create_polygons(num_polygons)
    min = 1104;
    max = 25206;
    counter = 1;
    plgs = cell(num_polygons, 2)
    while counter <= num_polygons
        rng('shuffle')
        numSides = ceil(8 + 42*rand(1, 1))
        [x, y, dt] = simple_polygon(numSides);
        area = polyarea(x,y)
        if area > min & area < max
            plgs{counter, 1} = x
            plgs{counter, 2} = y
            counter = counter + 1
        end
    end
    formatSpec = '%3.1f %3.1f;';
    fileID = fopen('polygons.txt', 'w');
    for index = 1:num_polygons
        [nrows, ~] = size(plgs{index, 1});
        fprintf(fileID, '[');
        for row = 1:nrows
            fprintf(fileID, formatSpec, plgs{index, 1}(row), plgs{index, 2}(row));
        end
        fprintf(fileID, ']');
    end
    fclose(fileID);
end
        