function [eff_TE,eff_TM] = RCWA_solver_arbitrary(wavelength,gap,thickness,acc,img_path)
    [prv,vmax]=retio([],inf*1i); % never write on the disc (nod to do retio)
    load('poly_Si.mat');
    n_medium = interp1(WL, R, wavelength)+1i*interp1(WL, I, wavelength);
    period = [gap,gap];% same unit as wavelength
    n_air = 1;% refractive index of the top layer
    n_glass = 1.5;% refractive index of the bottom layer
    angle_theta = 0;
    k_parallel = n_air * sin(angle_theta*pi/180);
    angle_delta = 0;
    parm = res0; % default parameters for "parm"
    parm.res1.champ = 1; % the eletromagnetic field is calculated accurately
    % parm.res1.trace = 1; % check the texture is right or not

    nn=[acc,acc];
    % textures for all layers including the top and bottom
    pixel_unit = gap/64;
    textures{1}= n_air; % uniform texture
    textures{2}= n_glass; % uniform texture
    textures{3} = shape_from_img(img_path,n_air,n_medium,pixel_unit);

    aa=res1(wavelength,period,textures,nn,k_parallel,angle_delta,parm);

    profile={[100,thickness,100],[2,3,1]};

    two_D=res2(aa,profile);
    eff_TE = two_D.TEinc_top_transmitted.efficiency;
    eff_TM = two_D.TMinc_top_transmitted.efficiency;
end

function texture = shape_from_img(img_path,n_air,n_medium,pixel_unit)
    img = imread(img_path);
    [width, height] = size(img);
    texture{1} = n_air;
    pixel = [];
    width_shift = width / 2;
    height_shift = height / 2;
    for i = - width_shift + 1:1:width_shift
        for j = -height_shift + 1:1:height_shift
            x = i + width_shift;
            y = j + height_shift;
            if img(x,y) == 0
                pixel = [pixel_unit*(i-0.5),pixel_unit*(j-0.5),pixel_unit,pixel_unit,n_air,1];
            else
                pixel = [pixel_unit*(i-0.5),pixel_unit*(j-0.5),pixel_unit,pixel_unit,n_medium,1]; 
            end
            texture = [texture,pixel];
        end
    end   
end
