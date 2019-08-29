spec=[];
for w=400:10:680
%Add Lumerical Matlab API path
path(path,'C:\Program Files\Lumerical\FDTD\api\matlab');
sim_file_path=('D:\FDTD\metaatom\metaatom\metaatom'); % update this path to user's folder
sim_file_name=('G_COLOR_Poly-Si.fsp');
 
%Open FDTD session
h=appopen('fdtd');



%Pass the path variables to FDTD
appputvar(h,'sim_file_path',sim_file_path);
appputvar(h,'sim_file_name',sim_file_name);

code=strcat('cd(sim_file_path);',...
    'load(sim_file_name);');      % open the selected file

appevalscript(h,code);
%groupscope("::model");
code=strcat('switchtolayout;',...
'groupscope("::model");',...   
'set("LENGTH_PERIOD",0.2*1e-6);',...  %set the period to 400 nm
'select("circle");',...
'set("radius",',num2str(0.08*1e-6),');',...  %set the radius 
'set("radius 2",',num2str(0.08*1e-6),');',... %set the radius 2 for ellipse metagrating
'set("z min",',num2str(-0.6*1e-6),');',...  %set the thickness
'select("source_R");',...
'set("wavelength start",',num2str(w*1e-9),');',... set wavelength
'set("wavelength stop",',num2str(w*1e-9),');',...
'select("source_R1");',...
'set("wavelength start",',num2str(w*1e-9),');',...
'set("wavelength stop",',num2str(w*1e-9),');',...
'run;');
appevalscript(h,code);               

%'set("index",3.5);',...                     %set the refractive index. It can be complex
code=strcat('Ey0=getdata("monitor","Ey");',...
    'Ex0=getdata("monitor","Ex");',...
    't=1;',...
    'sizey=size(Ey0);',...
    'tt=sizey(1,3);',...
    'l=tt;',...
    'cc=mean(abs(0.5*(Ex0(:,1,l,t)-1i*Ey0(:,1,l,t))));',...
    'T=transmission("monitor_2");');   %calculate results 



appevalscript(h,code);
cc=appgetvar(h,'cc');  %cross-polarization efficiency
T=appgetvar(h,'T');   %transmission
spec=[spec,T];

end
plot(spec)
