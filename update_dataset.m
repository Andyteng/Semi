function [ train_new, unlabeled_new, label_unew ] = update_dataset(index,train,unlabeled,label_u )


train_new = [train; unlabeled(index,:)];
unlabeled(index,:) = [];
unlabeled_new = unlabeled;

label_u(index) = [];
label_unew = label_u;

end
