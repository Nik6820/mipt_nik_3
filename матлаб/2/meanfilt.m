function y = meanfilt(x, win)
%MEANFILT  Простое скользящее среднее (SMA)
    x = x(:);
    n = numel(x);
    y = zeros(n,1);
    h = floor(win/2);
    for i = 1:n
        i1 = max(1, i-h);
        i2 = min(n, i+h);
        y(i) = mean(x(i1:i2));
    end
end