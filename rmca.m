function y = rmca(X, C, varargin) % C : sensitivity of slack variables

    % initialize
    dim = size(X, 1);
    n = size(X, 2);
    
    if C < 1
        error('C is too small, C means sensitivity of slack variables.');
    elseif C >= 2
        fprintf('Warning: the problem can be overtrained.\n');
    end
    
    C = C / n;
    
    % compute correlation matrix
    if nargin==3
        switch varargin{1}
            case 'linear'
                D = X' * X;
            case 'poly'
                D = (X' * X + 1)^2;
            case 'gauss'
                D = zeros(n,n);
                for i=1:n
                    for j=i:n
                        D(i,j) = exp(-norm(X(:,i) - X(:,j))^2);
                        D(j,i) = D(i,j);
                    end
                end
            otherwise
                for i = 1:n
                    X(:,i) = X(:,i) / norm(X(:,i),2);
                end
                D = X' * X;
        end
        O = ones(n)/n;
        D = D - O*D - D*O + O*D*O;
    else
        % normalize
        %X = X - repmat(mean(X),dim,1);
        for i = 1:n
            X(:,i) = X(:,i) / norm(X(:,i),2);
        end
        D = X' * X;
    end
    
    % solve duam problem
    cvx_quiet(true);
    cvx_begin
        variables a(n);
        a >= 0;
        a <= C;
        sum(a) == 1; %sum(a) == C*n - 1;
        minimize quad_form( a , D ); %minimize quad_form( C - a , D ); % (C-a)' * D * (C-a)
    cvx_end
    %y = sum( X .* repmat((C-a)',dim,1), 2 );
    y = X * a;
    
    if nargin==3
        switch varargin{1}
            case {'linear','poly','gauss'}
                norms = zeros(1,n);
                for i = 1:n
                    norms(i) = norm(X(:,i),2);
                end
                mean(norms)
                y = y / norm(y) * mean(norms);
            otherwise
                y = y / norm(y);
        end
    else
        y = y / norm(y);
    end
    
end