clear all
datatype=1;
doplot=1;
K=3;
%% Iris data
if datatype==1
    data=importIris;
    class(1:50)=1;
    class(51:100)=2;
    class(101:150)=3;
end
%% wine data
if datatype==2
    data = importdata('wine.data', ',',0);
    class=data(:,1);
    data(:,1)=[];
end
%% PART 3 centering
centre=1;
if centre==1
    data=centering(data);
end


%% PART 2 pca
% find covariance matrix of data, and compute eigenvalues and vectors
% covmat=cov(data);
covmat=1/(size(data,1))*(data'*data);
[vect val]=eig(covmat);
val=diag(val);

% find 2 largest eigenvalues and corresponding vectors
maxeig1=max(val);
ind1=find(val==max(val));
val(ind1)=0;
maxeig2=max(val);
ind2=find(val==max(val));

% put largest eigenvectors in a matrix
evect(:,1)=vect(:,ind1);
evect(:,2)=vect(:,ind2);

% multiply matrix of eigenvectors by data, and cluster centres to project
% data into new feature space
pcafeatures=evect'*data';

% pcamu=evect'*mu';

%% PART 2 K means

clear data
data=pcafeatures';
% initialise cluster centres as random values between the min and max data
% values
for p=1:size(data,2)
    a(p)=min(data(:,p)); b(p)=max(data(:,p));
    mu(:,p)=a(p)+(b(p)-a(p))*rand(K,1);
end

nochange=0;

while nochange==0
    % find euclidean distance between all points and the cluster centres
    % and assign points to closest cluster centres
    for i=1:size(data,1)
        for ii=1:size(mu,1)
            V=data(i,:)-mu(ii,:);
            dist(ii)=sqrt(V*V');
        end
        k(i)=find(dist==min(dist));
        clear dist
    end
    % if cluster centre is lost, reinitialise
    if length(unique(k))<K
        for p=1:size(data,2)
            mu(:,p)=a(p)+(b(p)-a(p))*rand(K,1);
        end
    else
        
        % update cluster centres according to points classified
        for pp=1:K
            if length(find(k==pp))==1
                tempmu(pp,:)=data(find(k==pp),:);
            else
                tempmu(pp,:)=mean(data(find(k==pp),:));
            end
        end        
        % check if the cluster centres have changed, if not end iterations
        if tempmu==mu
            nochange=1;
        else
            mu=tempmu;
        end
        clear tempmu
    end
end

% find the maximum score the clustering has acheived
conf=confusionmat(class, k');
comb=perms(1:size(conf,1));
for ii=1:size(comb,1)
    diagsum=0;
    for jj=1:size(comb,2)
        diagsum=diagsum+conf(comb(ii,jj),jj);
    end
    scores(ii)=diagsum;
    %/sum(sum(conf));
end
maxscore=max(scores)/sum(sum(conf))
comparray(1,:)=class;
comparray(2,:)=k';

for ppp=1:K
    pca{ppp}=pcafeatures(:,find(k==ppp));
end
pcamu=mu';
%% plot k means clustering after pca
if doplot==1
    figure;
    hold on
    col={'rx','gx','bx','kx','mx'};
    col2={'r*','g*','b*','k*','m*'};
    for P=1:K
        
        plot(pca{P}(1,:),pca{P}(2,:),col{P})
        hold on
        h=plot(pcamu(1,P), pcamu(2,P),col2{P});
        set(h, 'Markersize',15);
    end
    xlabel('Feature 1','fontsize', 12)
    ylabel('Feature 2','fontsize', 12)
    legend('cluster 1','centre 1','cluster 2','centre 2', 'cluster 3','centre 3', 'cluster 4','centre 4', 'cluster 5','centre 5')
end