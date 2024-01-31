function listcompare_spezial(list1,list2)

name1 = 'list1';
name2 = 'list2';
found = 0;
if isstr(list1)
    name1 = list1(end-11:end);
    [dummy1, dummy2, raw] = xlsread(list1,'LONGLIST');
    iend = find(diff(isnan([raw{3:end,10}])))+2;
    out= raw(1:iend,1:21);
    out(1,:) = strrep(out(1,:),'/','_');
    list1 = cell2struct(out,out(1,:),2);
    list1 = list1(3:end);
end

if isstr(list2)
    name2 = list2(end-11:end);
    [dummy1, dummy2, raw] = xlsread(list2,'LONGLIST');
    iend = find(diff(isnan([raw{3:end,10}])))+2;
    out= raw(1:iend,1:21);
    out(1,:) = strrep(out(1,:),'/','_');
    list2 = cell2struct(out,out(1,:),2);
    list2 = list2(3:end);
end

% remove all bendmarkers
    ibmark = strcmp('BENDMARK',{list1.TYPE});
    list1 = list1(find(~ibmark));
        ibmark = strcmp('BENDMARK',{list2.TYPE});
    list2 = list2(find(~ibmark));

ifound = ones(length(list2),1); 

for i = 1: length(list1)
    isection = nameparser('SECTION',list1(i).SECTION,list2);
    % find elements of list2 that are in the vincinity of actual element
    %[ds,j] =  sort(abs([list2(isection).Z]-list1(i).Z));
    [ds,j] =  sort(abs(([list2(isection).Z]-list1(i).Z)+([list2(isection).Y]-list1(i).Y)+([list2(isection).X]-list1(i).X)));
    for ij = 1:min(length(j),5)
        i2=isection(j(ij));
        if strcmp(list1(i).TYPE,list2(i2).TYPE) == 1
            ifound(i2) = 0;
            if ds(ij) < 1e-4
                if (strcmp(list1(i).NAME1,list2(i2).NAME1) && strcmp(list1(i).NAME2,list2(i2).NAME2)) == 1
                    list1ident(i, :) = [i i2 0 list1(i).Z-list2(i2).Z];
                elseif strcmp(list1(i).NAME1,list2(i2).NAME1) ~= 1
                    list1ident(i, :) = [i i2 -1 list1(i).Z-list2(i2).Z];
                elseif strcmp(list1(i).NAME2,list2(i2).NAME2) ~= 1
                    list1ident(i, :) = [i i2 -2 list1(i).Z-list2(i2).Z];
                end
                %                   disp([num2str(i) ':' list1(i).NAME1 ' unchanged' ]);
            else
                list1ident(i,:) = [i i2 1 list1(i).Z-list2(i2).Z];
                %                   disp([num2str(i) ':' list1(i).NAME1 ' moved by ' num2str(list1(i).S-list2(i2).S)]);
            end
            found = 1;
            break;
        end
    end
    if found == 0
        %          disp([num2str(i) ':' list1(i).NAME1 ' inserted '])
        list1ident(i,:) = [i 0 2 list1(i).Z];
    else
        found = 0;
    end
end
ifound = find(ifound==1);

fid = fopen('test.txt','w');

fprintf(fid,'%12s %12s %12s %12s %12s %10s %10s %10s %10s %10s %10s %10s %10s \n', ...
    'Section',name1,name1,name2,name2,'_','DeltaS','DeltaX','DeltaY','DeltaZ','DeltaTHETA','DeltaPHI','DeltaCHI');

ik = 1;
for i=1:length(list1ident)
    if ik <= length(ifound)
        if list1(i).S < list2(ifound(ik)).S
            
        else
            fprintf(fid,'%12s %i %10s %i %10s %10s %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f \n',...
                '_',0,'_',ifound(ik),list2(ifound(ik)).NAME1, ' removed',[],[],[],[],[],[],[]);
            ik=ik+1;
            
        end
    end
    switch list1ident(i,3)
        case 0
            fprintf(fid,'%12s %i %10s %i %10s %10s %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f \n',...
               list1(i).SECTION, i,list1(i).NAME1,list1ident(i,2),list2(list1ident(i,2)).NAME1, 'unchanged',[],[],[],[],[],[],[]);
        case -1   
            fprintf(fid,'%12s %i %10s %i %10s %10s %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f \n',...
               list1(i).SECTION, i,list1(i).NAME1,list1ident(i,2),list2(list1ident(i,2)).NAME1, 'NAME1_changed',[],[],[],[],[],[],[]);
        case -2
            fprintf(fid,'%12s %i %10s %i %10s %10s %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f \n',...
               list1(i).SECTION, i,list1(i).NAME2,list1ident(i,2),list2(list1ident(i,2)).NAME2, 'NAME2_changed',[],[],[],[],[],[],[]);
        case 1
            fprintf(fid,'%12s %i %10s %i %10s %10s %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f \n',...
                list1(i).SECTION,i,list1(i).NAME1,list1ident(i,2),list2(list1ident(i,2)).NAME1, ' moved_by ',list1(i).S-list2(list1ident(i,2)).S, ...
                list1(i).X-list2(list1ident(i,2)).X,list1(i).Y-list2(list1ident(i,2)).Y,list1(i).Z-list2(list1ident(i,2)).Z,...
                list1(i).THETA-list2(list1ident(i,2)).THETA,list1(i).PHI-list2(list1ident(i,2)).PHI,list1(i).CHI-list2(list1ident(i,2)).CHI);
        case 2
            fprintf(fid,'%12s %i %10s %i %10s %10s %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f %2.6f \n',...
                list1(i).SECTION,i,list1(i).NAME1,0,'_', 'inserted',[],[],[],[],[],[],[]);
            
    end
end

fclose(fid)