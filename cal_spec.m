function [TE,TM] = cal_spec(gap,thickness,acc,img_path)
    TE = [];
    TM = [];
    for w=400:10:680 % parfor must be continuous
        [eff_TE, eff_TM] = RCWA_solver_arbitrary(w,gap,thickness,acc,img_path);
        TE = [TE,eff_TE];
        TM = [TM,eff_TM];
        
    end
end