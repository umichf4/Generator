acc = 5;
times = 1;
T = [];
for gap = 200:5:400
    for thickness = 200:50:700
        for radius = 20:5:90
            for wavelength = 400:10:680
                eff = RCWA_solver(wavelength,gap,thickness,radius,acc);
                T(times,1) = wavelength;
                T(times,2) = thickness;
                T(times,3) = radius;
                T(times,4) = gap;
                T(times,5) = eff;
                times = times + 1;
                disp(100 * times / 196185);
            end
        end
    end
end

save 'circle_0901.mat' T