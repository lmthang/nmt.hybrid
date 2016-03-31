function install_cppcores()
% mex ALL cpp cores in sltoolbox
% Note those relying on third-party libraries are not involved here
% such as annsearch, which need to be mex separately.

% core\pwcalc_core.cpp
% core\pwdiff_core.cpp
% core\rowcolop_core.cpp
% core\vecop_core.cpp
% discrete\histmetricpw_core.cpp
% utils_ex\win32guid_core.cpp

srcs = { ...
    'core', 'pwcalc_core.cpp'; ...
    'core', 'pwdiff_core.cpp'; ...
    'core', 'rowcolop_core.cpp'; ...
    'core', 'vecop_core.cpp'; ...
    'discrete', 'histmetricpw_core.cpp'; ...
    'utils_ex', 'win32guid_core.cpp'};
    
n = size(srcs, 1);
for i = 1 : n
   subdir = srcs{i, 1};
   srcfn = srcs{i, 2};
   
   fprintf('Processing %s\\%s ...\n', subdir, srcfn);
   
   cd(subdir);
   mex('-O', srcfn);
   cd('..');

end
    