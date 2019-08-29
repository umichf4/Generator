eff_list = [];
eff = zeros(1,4);
times = 0;
for w = 400:10:680
    for thickness = -0.056
        for r = -0.366
            eff(1,1) = w;
            eff(1,2) = 360;
            eff(1,3) = thickness;
            eff(1,4) = r;
            eff(1,5) = RCWA_solver(w, 360, thickness, r, 6);
            eff_list = [eff_list; eff];
            times = times + 1;
            progress = times / (29*16*9)
        end
    end
end

   