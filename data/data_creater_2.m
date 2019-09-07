acc = 5;
gap_all = 200:5:400;
thickness_all = 200:50:700;
radius_all = 20:5:90;
wavelength_all = 400:10:680;
number = size(gap_all, 2) * size(radius_all, 2) * size(wavelength_all, 2) * size(thickness_all, 2);

wavelength_row = repmat(wavelength_all', size(gap_all, 2) * size(thickness_all, 2) * size(radius_all, 2), 1);

thickness_row_temp = repmat(thickness_all, size(gap_all, 2) * size(radius_all, 2) * size(wavelength_all, 2), 1);
thickness_row = thickness_row_temp(:);

radius_row_temp = repmat(radius_all, size(gap_all, 2) * size(wavelength_all, 2), 1);
radius_row_temp2 = repmat(radius_row_temp(:), 1, size(thickness_all, 2));
radius_row = radius_row_temp2(:);

gap_row_temp = repmat(gap_all, size(wavelength_all, 2), 1);
gap_row_temp2 = repmat(gap_row_temp(:), 1, size(thickness_all, 2) * size(radius_all, 2));
gap_row = gap_row_temp2(:);

eff_row = zeros(number, 1);

T = cat(2, wavelength_row, thickness_row, radius_row, gap_row, eff_row);
result = zeros(number, 1);

parfor index = 1:1:number
    result(index) = RCWA_solver(T(index, 1), T(index, 4), T(index, 2), T(index, 3), acc);
    disp(100 * index / 196185);
end

T(:,5) = result;

save 'circle_0901.mat' T