function eff = RCWA_solver(wavelength,gap,thickness,radius,acc,medium,shape)
    [prv,vmax]=retio([],inf*1i); % never write on the disc (nod to do retio)
    if medium == 0
        load('poly_Si.mat');
    elseif medium == 1
        load('TiO2.mat');
    end
    n_medium = interp1(WL, R, wavelength)+1i*interp1(WL, I, wavelength);
    period = [gap,gap];% same unit as wavelength
    n_air = 1;% refractive index of the top layer
    n_glass = 1.5;% refractive index of the bottom layer
    angle_theta = 0;
    k_parallel = n_air*sin(angle_theta*pi/180);
    angle_delta = 0;
    parm = res0; % default parameters for "parm"
    parm.sym.pol = 1; % TE
    parm.res1.champ = 1; % the eletromagnetic field is calculated accurately
    parm.sym.x=0;parm.sym.y=0;% use of symetry

    nn=[acc,acc];
    % textures for all layers including the top and bottom
    texture = cell(1,3);
    textures{1}= n_air; % uniform texture
    textures{2}= n_glass; % uniform texture
    if shape == 0
        textures{3}={n_air,[0,0,radius*2,radius*2,n_medium,10] };
    elseif shape == 1
        textures{3}={n_air,[0,0,radius*2,radius*2,n_medium,1] };
    end
    aa=res1(wavelength,period,textures,nn,k_parallel,angle_delta,parm);

    profile={[100,thickness,100],[2,3,1]};

    two_D=res2(aa,profile);
    eff = two_D.TEinc_top_transmitted.efficiency_TE;
end
