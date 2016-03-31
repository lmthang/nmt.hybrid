function display(DS)
%DISPLAY displays the dataset fields.

if isequal(get(0,'FormatSpacing'),'compact')
    disp([inputname(1) ' =']);
    disp(DS);
else
    disp(' ');
    disp([inputname(1) ' =']);
    disp(' ');
    disp(DS);
end