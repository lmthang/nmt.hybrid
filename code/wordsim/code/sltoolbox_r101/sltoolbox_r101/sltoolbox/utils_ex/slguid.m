function gid = slguid()
%SLGUID Generates a GUID (Global Unique Identifier)
%
% $ Syntax $
%   - gid = slguid()
%
% $ Description $
%   - gid = slguid() generates a 128-bit GUID, which is represented by
%     a 16 x 1 UINT8 array. If it succeeds, an 1 x 16 array is returned
%     otherwise an error is reported.
%
% $ Remarks $
%   - In current implementation, it is based on win32guid_core mex
%     cpp function, thus only Win32 or compatible platform is supported.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12, 2006
%

if ~strcmpi(computer(), 'PCWIN') && ~strcmpi(computer(), 'PCWIN64')
    error('sltoolbox:notsupportplatform', ...
        'Only win32 or x64 platform is supported for slguid');
end

[b, gid] = win32guid_core();

if ~b
    error('sltoolbox:rterror', ...
        'Fail to generate GUID');
end

    


