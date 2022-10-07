clear all;

% set the seed
rand('seed',6);

% set the number of iterations
REP=200;

% N: total sample size, m: number of machines, n: local sample size, p: dimension, T: number of iterations
N=10000;
m=100;
n=N/m;
p=6;
T=200;

% fraction of Byzantine machines
alpha=0.2;

% threshold parameter for the trimmed mean
xi=0.2;

% set parameters
lambda=0.01;
eta=1/(sqrt(p)+lambda);

% true beta for the model
for i=1:p
    for j=1:p
        Sig(i,j)=0.4^(abs(i-j));
    end
end
beta_true = [1.1*ones(4,1);zeros(p-4,1)];

for rep=1:REP
    randn('state',rep)
    
    % compute the initial estimator
    x0 = mvnrnd(zeros(p,1),Sig,n);
    y0 = 2*binornd(1,normcdf(x0*beta_true),n,1)-1; 
    cvx_begin quiet
    variable b(p,1)
    minimize sum(max(1-y0.*(x0*b),0))
    cvx_end
    
    % estimate of vanilla distributed gradient descent algorithm without Byzantine attacks
    beta_mean0 = b;
    % estimate of vanilla distributed gradient descent algorithm under moderate attacks
    beta_mean_mo = b;
    % estimate of median-based distributed gradient descent algorithm under moderate attacks
    beta_median_mo = b;
    % estimate of trimmed-mean-based distributed gradient descent algorithm under moderate attacks
    beta_trim_mo = b;
    % estimate of vanilla distributed gradient descent algorithm under extreme attacks
    beta_mean_ex = b;
    % estimate of median-based distributed gradient descent algorithm under extreme attacks
    beta_median_ex = b;
    % estimate of trimmed-mean-based distributed gradient descent algorithm under extreme attacks
    beta_trim_ex = b;
    
    % generate data
    x = mvnrnd(zeros(p,1),Sig,N);
    y = 2*binornd(1,normcdf(x*beta_true),N,1)-1; 
    
    for t=1:T
        for k=1:m
            ym=y(((k-1)*n+1):(k*n),:);
            xm=x(((k-1)*n+1):(k*n),:);
            G_mean0(k,:)=-1/m*sum(xm.*repmat(ym.*max(sign(1-ym.*(xm*beta_mean0)),0),1,p))+lambda*beta_mean0';
            G_mean_mo(k,:)=-1/m*sum(xm.*repmat(ym.*max(sign(1-ym.*(xm*beta_mean_mo)),0),1,p))+lambda*beta_mean_mo';
            G_median_mo(k,:)=-1/m*sum(xm.*repmat(ym.*max(sign(1-ym.*(xm*beta_median_mo)),0),1,p))+lambda*beta_median_mo';
            G_trim_mo(k,:)=-1/m*sum(xm.*repmat(ym.*max(sign(1-ym.*(xm*beta_trim_mo)),0),1,p))+lambda*beta_trim_mo'; 
            G_mean_ex(k,:)=-1/m*sum(xm.*repmat(ym.*max(sign(1-ym.*(xm*beta_mean_ex)),0),1,p))+lambda*beta_mean_ex';
            G_median_ex(k,:)=-1/m*sum(xm.*repmat(ym.*max(sign(1-ym.*(xm*beta_median_ex)),0),1,p))+lambda*beta_median_ex';
            G_trim_ex(k,:)=-1/m*sum(xm.*repmat(ym.*max(sign(1-ym.*(xm*beta_trim_ex)),0),1,p))+lambda*beta_trim_ex'; 
        end
        
        % moderate random attack
        G_mean_mo(1:(m*alpha),:)=unifrnd (-1,1,m*alpha,p);
        G_median_mo(1:(m*alpha),:)=unifrnd (-1,1,m*alpha,p);
        G_trim_mo(1:(m*alpha),:)=unifrnd (-1,1,m*alpha,p);
        % extreme random attack
        G_mean_ex(1:(m*alpha),:)=unifrnd (-10,10,m*alpha,p);
        G_median_ex(1:(m*alpha),:)=unifrnd (-10,10,m*alpha,p);
        G_trim_ex(1:(m*alpha),:)=unifrnd (-10,10,m*alpha,p);
        
        g_mean0 = mean(G_mean0);
        g_mean_mo = mean(G_mean_mo);
        g_median_mo = median(G_median_mo);
        g_trim_mo = trimmean(G_trim_mo,2*xi*100);
        g_mean_ex = mean(G_mean_ex);
        g_median_ex = median(G_median_ex);
        g_trim_ex = trimmean(G_trim_ex,2*xi*100);
        
        % update the estimates
        beta_mean0 = beta_mean0-eta*g_mean0';
        beta_mean_mo = beta_mean_mo-eta*g_mean_mo';
        beta_median_mo = beta_median_mo-eta*g_median_mo';
        beta_trim_mo = beta_trim_mo-eta*g_trim_mo';
        beta_mean_ex = beta_mean_ex-eta*g_mean_ex';
        beta_median_ex = beta_median_ex-eta*g_median_ex';
        beta_trim_ex = beta_trim_ex-eta*g_trim_ex';
        
        err_mean0(rep,t) = norm(beta_mean0-beta_true);
        err_mean_mo(rep,t) = norm(beta_mean_mo-beta_true);
        err_median_mo(rep,t) = norm(beta_median_mo-beta_true);
        err_trim_mo(rep,t) = norm(beta_trim_mo-beta_true);
        err_mean_ex(rep,t) = norm(beta_mean_ex-beta_true);
        err_median_ex(rep,t) = norm(beta_median_ex-beta_true);
        err_trim_ex(rep,t) = norm(beta_trim_ex-beta_true);
    end   
end
% compute errors
E_mean0=mean(err_mean0);
E_mean_mo=mean(err_mean_mo);
E_median_mo=mean(err_median_mo);
E_trim_mo=mean(err_trim_mo);
E_mean_ex=mean(err_mean_ex);
E_median_ex=mean(err_median_ex);
E_trim_ex=mean(err_trim_ex);

% output the final errors
[E_mean0(T), E_mean_mo(T), E_median_mo(T), E_trim_mo(T), E_mean_ex(T), E_median_ex(T), E_trim_ex(T)]