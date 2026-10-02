function mse = calcMSE(orig, filt)
% CALCMSE Среднеквадратичная ошибка между двумя изображениями
    orig = double(orig);
    filt = double(filt);
    diff = orig - filt;
    mse  = mean(diff(:).^2);
end