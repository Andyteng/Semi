function [err] = cal_err(lab,lab_pred)

err = sum(lab~=lab_pred)/length(lab);

