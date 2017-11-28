function data=centering(data,stddev)
% subtract the mean of each column by each data point in that column
for p=1:size(data,2)
    
    data(:,p)=data(:,p)-mean(data(:,p));
    if stddev==1
    data(:,p)=data(:,p)/std(data(:,p));
    end
end