function eff_list = RCWA_solver_par(wavelength,gap,thickness,radius,acc)
eff_list = [];
len = length(wavelength);

parfor index = 1:1:len
    eff_list(index) = RCWA_solver(wavelength(index),gap,thickness,radius,acc);
end

end
