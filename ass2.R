train <- read.csv("C:/Users/sindr/Google Drive/Dokumenter/NTNU/Emner/Machine learning/Assignements/Ass2/train.csv")
test <- read.csv("C:/Users/sindr/Google Drive/Dokumenter/NTNU/Emner/Machine learning/Assignements/Ass2/test.csv")
#which(is.na(train))

library(glmnet)
x = sparse.model.matrix(Activity~.-1,train)
y = train[,1]

#perform grid search to find optimal value of lambda
#family= binomial => logistic regression, alpha=1 => lasso
# check docs to explore other type.measure options
cv.out <- cv.glmnet(x,y,alpha=1,family="binomial",type.measure="deviance")

#plot result
plot(cv.out)

lambda_1se = cv.out$lambda.1se
coefs = coef(cv.out,s=lambda_1se)
fit = cv.out$glmnet.fit

results = data.frame(
  ds = which(coefs!=0),
  cs = coefs[which(coefs!=0)]
)

int_cols = sapply(train,class)=="integer"
int_col_vals = train[,int_cols]
num_cols = sapply(train,class)=="numeric"

min_var = 1000
max_var = 0
for (i in which(num_cols)){
  this_var[i] = var(train[,i])
  if(this_var[i] < min_var){
    min_var = this_var[i]
    print(i)
  }
  if(this_var[i] > max_var)
    max_var = this_var[i]
}

j = 1
for (i in which(num_cols)){
  non_nulls[j] = sum(train[,i]!=0)
  j = j+1
}

k = 1
for (j in which(num_cols)){
  if (non_nulls[k] <= 10){
    print('column')
    print(j)
    for (i in 1:3751){
      if (train[i,which(num_cols)[k]]!=0){
        print(train[i,which(num_cols)[k]])
      }
    }
  }
  k = k+1
}
