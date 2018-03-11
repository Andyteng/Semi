% ????????????  ?ML???  
  
clc;  
clear;  
  
% ???????????  
Mu = [0 0; 3 3]';  
% ?????  
S1 = 0.8 * eye(2);  
S(:, :, 1) = S1;  
S(:, :, 2) = S1;  
% ??????????  
P = [1/3 2/3]';  
% ??????  
% ?????????????????????????????  
N = 500;  
  
  
% 1.?????????  
%{  
    ??????  
    N = 500,  c = 2, d = 2  
    ?1=[0, 0]'   ?2=[3, 3]'  
    S1=S2=[0.8, 0; 0.8, 0]  
    p(w1)=1/3   p(w2)=2/3  
%}  
randn('seed', 0);  
[X_train, Y_train] = generate_gauss_classes(Mu, S, P, N);  
  
figure();  
hold on;  
class1_data = X_train(:, Y_train==1);  
class2_data = X_train(:, Y_train==2);  
plot(class1_data(1, :), class1_data(2, :), 'r.');  
plot(class2_data(1, :), class2_data(2, :), 'g.');  
grid on;  
title('????');  
xlabel('N=500');  
  
%{  
    ????????????  
    N = 500,  c = 2, d = 2  
    ?1=[0, 0]'   ?2=[3, 3]'  
    S1=S2=[0.8, 0; 0.8, 0]  
    p(w1)=1/3   p(w2)=2/3  
%}  
randn('seed', 100);  
[X_test, Y_test] = generate_gauss_classes(Mu, S, P, N);  
figure();  
hold on;  
test1_data = X_test(:, Y_test==1);  
test2_data = X_test(:, Y_test==2);  
plot(test1_data(1, :), test1_data(2, :), 'r.');  
plot(test2_data(1, :), test2_data(2, :), 'g.');  
grid on;  
title('????');  
xlabel('N=500');  
  
  
% 2.???????ML??????  
% ??????????????????????????????????  
[mu1_hat, s1_hat] = gaussian_ML_estimate(class1_data);  
[mu2_hat, s2_hat] = gaussian_ML_estimate(class2_data);  
mu_hat = [mu1_hat, mu2_hat];  
s_hat = (1/2) * (s1_hat + s2_hat);  
  
  
% 3.????????????????  
% ??????????  
z_euclidean = euclidean_classifier(mu_hat, X_test);  
% ???????????  
z_bayesian = bayes_classifier(Mu, S, P, X_test);  
  
  
% 4.???????????  
err_euclidean = ( 1-length(find(Y_test == z_euclidean')) /  length(Y_test) );  
err_bayesian = ( 1-length(find(Y_test == z_bayesian')) /  length(Y_test) );  
% ????  
disp(['?????????????', num2str(err_euclidean)]);  
disp(['?????????????????', num2str(err_bayesian)]);  
  
  
% ????  
figure();  
hold on;  
z_euclidean = transpose(z_euclidean);  
o = 1;  
q = 1;  
for i = 1:size(X_test, 2)  
    if Y_test(i) ~= z_euclidean(i)  
        plot(X_test(1,i), X_test(2,i), 'bo');  
    elseif z_euclidean(i)==1  
        euclidean_classifier_results1(:, o) = X_test(:, i);  
        o = o+1;  
    elseif z_euclidean(i)==2  
        euclidean_classifier_results2(:, q) = X_test(:, i);  
        q = q+1;  
    end  
end  
plot(euclidean_classifier_results1(1, :), euclidean_classifier_results1(2, :), 'r.');  
plot(euclidean_classifier_results2(1, :), euclidean_classifier_results2(2, :), 'g.');  
title(['??????????????', num2str(err_euclidean)]);  
grid on;  
  
figure();  
hold on;  
z_bayesian = transpose(z_bayesian);  
o = 1;  
q = 1;  
for i = 1:size(X_test, 2)  
    if Y_test(i) ~= z_bayesian(i)  
        plot(X_test(1,i), X_test(2,i), 'bo');  
    elseif z_bayesian(i)==1  
        bayesian_classifier_results1(:, o) = X_test(:, i);  
        o = o+1;  
    elseif z_bayesian(i)==2  
        bayesian_classifier_results2(:, q) = X_test(:, i);  
        q = q+1;  
    end  
end  
plot(bayesian_classifier_results1(1, :), bayesian_classifier_results1(2, :), 'r.');  
plot(bayesian_classifier_results2(1, :), bayesian_classifier_results2(2, :), 'g.');  
title(['?????????????????????', num2str(err_bayesian)]);  
grid on; 