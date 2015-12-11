function splittedstring = strsplit(inpstr,delimiter)
%FUNCTION strsplit
%
%-USE:
%
%This function should be used to split a string of delimiter separated
%values.  If all values are numerical values the returned matrix is a
%double array but if there is one non numerical value a cell array is
%returned.  You can check this with the iscell() function.
%
%-SYNTAX
%output = strsplit(inpstr[,delimiter])
%
%inpstr:    string containing delimiter separatede numerical values, eg
%           3498,48869,23908,34.67
%delimiter: optional, if omitted the delimiter is , (comma)
%
%-OUTPUT
%
%An x by 1 matrix containing the splitted values
%
%-INFO
%
%mailto:    gie.spaepen@ua.ac.be
%%%%%%%%%%%%
% changed by richard@socher.org to not return empty strings inside the cell
% and to make a returned string into a 1d cell, so it's consistent
%!!!!
% Alternative with regular expressions: splits by all tab characters, output is in splitStrings
%     [matchStart, matchEnd, tokenIndices, matchStrings, tokenStrings, tokenName, splitStrings] = regexp(stringInput, '\t')
%--------------------------------------------------------------------------



%Check input arguments
if(nargin < 1)
    error('There is no argument defined');
else
    if(nargin == 1)
        strdelim = ',';
        %Verbose off!! disp 'Delimiter set to ,';
    else
        strdelim = delimiter;
    end
end

%deblank string
deblank(inpstr);

%Get number of substrings
idx  = findstr(inpstr,strdelim);
if size(idx) == 0
    %     disp 'No delimiter in string, inputString is returned';
    splittedstring = str2double(inpstr); % Thang fix Nov 27, 2013, splittedstring = inpstr;
else
    %Define size of the indices
    sz = size(idx,2);
    %Define splittedstring
    tempsplit = {};
    %Loop through string and itinerate from delimiter to delimiter
    for i = 1:sz
        %Define standard start and stop positions for the start position,
        %choose 1 as startup position because otherwise you get an array
        %overflow, for the endposition you can detemine it from the
        %delimiter position
        strtpos = 1;
        endpos = idx(i)-1;
        %If i is not the beginning of the string get it from the delimiter
        %position
        if i ~= 1
            strtpos = idx(i-1)+1;
        end
        %If i is equal to the number of delimiters get the last element
        %first by determining the lengt of the string and then replace the
        %endpos back to a standard position
        if i == sz
            endpos = size(inpstr,2);
            tempsplit(i+1) = {inpstr(idx(i)+1 : endpos)};
            endpos = idx(i)-1;
        end
        %Add substring to output: splittedstring a cell array
        tempsplit(i) = {inpstr(strtpos : endpos)};
    end
    %Flag
    isallnums = 1;
    %Check is there are NaN values if matrix elements are converted to
    %doubles
    for i = 1:size(tempsplit,2)
        tempdouble = str2double(tempsplit(i));
        if(isnan(tempdouble))
            isallnums = 0;
        end
    end
    %If isallnums = 1 then return a double array otherwise return a cell
    %array
    if(isallnums == 1)
        for i = 1:size(tempsplit,2)
            splittedstring(i) = str2double(tempsplit(i));
        end
    else
        splittedstring = tempsplit;
    end
    
    
end

if iscell(splittedstring) && length(splittedstring)>1
    for s = 1:length(splittedstring)
        if isempty(splittedstring{s})
            splittedstring(s) = [];
        end
    end
end

if isstr(splittedstring) && length(splittedstring)
    splittedstring={splittedstring};
end
