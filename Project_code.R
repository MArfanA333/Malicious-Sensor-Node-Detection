########################################################################
library(dplyr)
library(ggplot2)
library(corrplot) 
library(caret)
library(corrr)
library(factoextra)
library(randomForest)
library(e1071)
library(ROCR)
library(pROC)
library(DMwR2)
library(smotefamily)
library(ROSE)
library(xgboost)

########################################################################
#Defined functions:
printMatrixInfo = function(confM, model){
  confM.accuracy = ((confM$table[1,1] + confM$table[2,2])/sum(confM$table))*100
  confM.precision = ((confM$table[2,2])/(confM$table[2,2] + confM$table[2,1]))*100
  confM.recall = ((confM$table[2,2] /(confM$table[1,2] + confM$table[2,2])))*100
  confM.F1 = 2 * ((confM.precision * confM.recall) / (confM.precision + confM.recall))
  cat("Accuracy of" ,model,": ", confM.accuracy, "%\n")
  cat("Precision of" ,model,": ", confM.precision, "%\n")
  cat("Recall of" ,model,": ", confM.recall, "%\n")
  cat("F1 score of" ,model,": ", confM.F1, "%\n")
}
printMatrixInfoXGBoost = function(confM, model){
  confM.accuracy = ((confM[1,1] + confM[2,2])/sum(confM))*100
  confM.precision = ((confM[2,2])/(confM[2,2] + confM[2,1]))*100
  confM.recall = ((confM[2,2] /(confM[1,2] + confM[2,2])))*100
  confM.F1 = 2 * ((confM.precision * confM.recall) / (confM.precision + confM.recall))
  cat("Accuracy of" ,model,": ", confM.accuracy, "%\n")
  cat("Precision of" ,model,": ", confM.precision, "%\n")
  cat("Recall of" ,model,": ", confM.recall, "%\n")
  cat("F1 score of" ,model,": ", confM.F1,"%\n")
}
# Importing data set:
setwd("C:/Users/User/Desktop")
rawSensorData = read.csv("sensornodes.csv")

# Modifying data set:

row.names(rawSensorData) = rawSensorData[,1]
rawSensorData = rawSensorData[,-1]
rawSensorData = distinct(rawSensorData)
rawSensorData$Is_Malicious <- as.factor(rawSensorData$Is_Malicious)
rawSensorData$Number_of_Neighbors <- as.factor(rawSensorData$Number_of_Neighbors)

# Checking for null values:
sum(is.na(rawSensorData))
dim(rawSensorData)

View(SensorData)
# Removing unneeded variables:
SensorData = rawSensorData[,c(-1,-2)]

SensorData.scaled = scale(SensorData[,c(-9,-18)])
SensorData.scaled = data.frame(SensorData.scaled,SensorData$Number_of_Neighbors,SensorData$Is_Malicious)

# Renaming Variables:
SensorData.scaled$Number_of_Neighbors = SensorData.scaled$SensorData.Number_of_Neighbors
SensorData.scaled = SensorData.scaled[, -which(colnames(SensorData.scaled) == "SensorData.Number_of_Neighbors")]
SensorData.scaled$Is_Malicious = SensorData.scaled$SensorData.Is_Malicious
SensorData.scaled = SensorData.scaled[, -which(colnames(SensorData.scaled) == "SensorData.Is_Malicious")]

# Checking the data:
summary(SensorData)
summary(SensorData.scaled)
cor(SensorData[,-c(9,18)])
print(var(SensorData))
zero_variance_vars <- nearZeroVar(SensorData)
print(zero_variance_vars)
dim(SensorData)

# Using SMOTE to balance the data set:

SensorData.scaled.ROSED = ROSE(Is_Malicious~.,data = SensorData.scaled,seed=93780)
summary(SensorData.scaled.ROSED)

set.seed(87520)
partition =  createDataPartition(SensorData.scaled.ROSED$data$Is_Malicious, p = 0.6, list = FALSE)

# Creating the training set:
train.SensorData.scaled.ROSED = SensorData.scaled.ROSED$data[partition,]

# Creating the testing set:
test.SensorData.scaled.ROSED = SensorData.scaled.ROSED$data[-partition, ]

# Creating PCA and plotting:
Matrix = cor(SensorData.scaled.ROSED$data[,c(-17,-18)])
corrplot(Matrix, method="color") 
SensorData.scaled.ROSED.pca = prcomp(SensorData.scaled.ROSED$data[, c(-17,-18)], scale. = TRUE)
summary(SensorData.scaled.ROSED.pca)

fviz_pca_var(SensorData.scaled.ROSED.pca, col.var = "cos2",
             gradient.cols = c("black", "orange", "green"),
             repel = TRUE, ggtheme = theme_minimal() + theme(aspect.ratio = 1))

# Scree Plot:
fviz_eig(SensorData.scaled.ROSED.pca, addlabels = TRUE)
fviz_cos2(SensorData.scaled.ROSED.pca, choice = "var", axes = 1:2)

########################################################################
# Splitting the data set(Train and test):
set.seed(93780)

partition =  createDataPartition(SensorData.scaled$Is_Malicious, p = 0.6, list = FALSE)

# Creating the training set:
train.SensorData = SensorData.scaled[partition,]

# Creating the testing set:
test.SensorData = SensorData.scaled[-partition, ]


# Choosing columns for the training and testing:

columns = c(3,4,18)
set.seed(87520)
train.SensorData.chosen = train.SensorData[,columns]
test.SensorData.chosen = test.SensorData[,columns]

summary(train.SensorData.chosen$Is_Malicious)
summary(test.SensorData.chosen$Is_Malicious)

# Random Forest Model:
# Finding the best number of random variables (m)
bestmtry <- tuneRF(train.SensorData.chosen, train.SensorData.chosen$Is_Malicious, stepFactor = 1.2, improve = 0.01, trace = T, plot = T)
print(bestmtry)



########################################################################

# Using SMOTE to balance data set and running RF and SVM:

# Choosing columns for the training and testing:

train.SensorData.chosen.ROSED = train.SensorData.scaled.ROSED[,columns]
test.SensorData.chosen.ROSED = test.SensorData.scaled.ROSED[,columns]
train.SensorData.chosen.ROSED = train.SensorData.chosen.ROSED[sample(nrow(train.SensorData.chosen.ROSED), size = 100), ]
summary(train.SensorData.chosen.ROSED$Is_Malicious)
summary(test.SensorData.chosen.ROSED$Is_Malicious)
predictors_only <- test.SensorData.chosen.ROSED[, names(test.SensorData.chosen.ROSED) != "Is_Malicious"]
response_only <- test.SensorData.chosen.ROSED$Is_Malicious
# RF:

SensorData.chosen.ROSED.RF = randomForest(Is_Malicious~., data = train.SensorData.chosen.ROSED, mtry=1)
SensorData.chosen.ROSED.RF.predicted = predict(SensorData.chosen.ROSED.RF,newdata = predictors_only,type="response")

SensorData.chosen.ROSED.RF.confM <- confusionMatrix(SensorData.chosen.ROSED.RF.predicted, test.SensorData.chosen.ROSED$Is_Malicious)
printMatrixInfo(SensorData.chosen.ROSED.RF.confM,"Random Forest Model(balanced data)")
probabilities <- predict(SensorData.chosen.ROSED.RF, newdata = test.SensorData.chosen.ROSED, type = "prob")[, 2]

roc_obj <- roc(response_only, probabilities)
plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)
auc_obj <- auc(roc_obj)
print(auc_obj)


# SVM:
SensorData.chosen.ROSED.SVM.radial = svm(Is_Malicious ~ ., data = train.SensorData.chosen.ROSED, kernel = "radial", type = 'C-classification')
SensorData.chosen.ROSED.SVM.radial.predicted = predict(SensorData.chosen.ROSED.SVM.radial,newdata = predictors_only,type="response")
summary(SensorData.chosen.ROSED.SVM.radial)
SensorData.chosen.ROSED.SVM.linear = svm(Is_Malicious ~ ., data = train.SensorData.chosen.ROSED, kernel = "linear", type = 'C-classification')
SensorData.chosen.ROSED.SVM.linear.predicted = predict(SensorData.chosen.ROSED.SVM.linear,newdata = predictors_only,type="response")
summary(SensorData.chosen.ROSED.SVM.linear)
SensorData.chosen.ROSED.SVM.poly = svm(Is_Malicious ~ ., data = train.SensorData.chosen.ROSED, kernel = "poly", type = 'C-classification')
SensorData.chosen.ROSED.SVM.poly.predicted = predict(SensorData.chosen.ROSED.SVM.poly,newdata = predictors_only,type="response")
summary(SensorData.chosen.ROSED.SVM.poly)
SensorData.chosen.ROSED.SVM.sigmoid = svm(Is_Malicious ~ ., data = train.SensorData.chosen.ROSED, kernel = "sigmoid", type = 'C-classification')
SensorData.chosen.ROSED.SVM.sigmoid.predicted = predict(SensorData.chosen.ROSED.SVM.sigmoid,newdata = predictors_only,type="response")
summary(SensorData.chosen.ROSED.SVM.sigmoid)
SensorData.chosen.ROSED.SVM.radial.confM <- confusionMatrix(SensorData.chosen.ROSED.SVM.radial.predicted, test.SensorData.chosen.ROSED$Is_Malicious)
printMatrixInfo(SensorData.chosen.ROSED.SVM.radial.confM, "SVM with radial kernal(balanced data)")

SensorData.chosen.ROSED.SVM.linear.confM <- confusionMatrix(SensorData.chosen.ROSED.SVM.linear.predicted, test.SensorData.chosen.ROSED$Is_Malicious)
printMatrixInfo(SensorData.chosen.ROSED.SVM.linear.confM, "SVM with linear kernal(balanced data)")

SensorData.chosen.ROSED.SVM.poly.confM <- confusionMatrix(SensorData.chosen.ROSED.SVM.poly.predicted, test.SensorData.chosen.ROSED$Is_Malicious)
printMatrixInfo(SensorData.chosen.ROSED.SVM.poly.confM, "SVM with poly kernal(balanced data)")

SensorData.chosen.ROSED.SVM.sigmoid.confM <- confusionMatrix(SensorData.chosen.ROSED.SVM.sigmoid.predicted, test.SensorData.chosen.ROSED$Is_Malicious)
printMatrixInfo(SensorData.chosen.ROSED.SVM.sigmoid.confM, "SVM with sigmoid kernal(balanced data)")

# Plotting all SVM Models:
plot(SensorData.chosen.ROSED.SVM.linear,SensorData.scaled.ROSED$data[,columns])
plot(SensorData.chosen.ROSED.SVM.radial,SensorData.scaled.ROSED$data[,columns])
plot(SensorData.chosen.ROSED.SVM.poly,SensorData.scaled.ROSED$data[,columns])
plot(SensorData.chosen.ROSED.SVM.sigmoid,SensorData.scaled.ROSED$data[,columns])

# XGBoost:

# Matrices for xgboost:

train.SensorData.xgboost.ROSED = train.SensorData.chosen.ROSED
test.SensorData.xgboost.ROSED = test.SensorData.chosen.ROSED

train.SensorData.xgboost.ROSED$Is_Malicious = as.numeric(train.SensorData.xgboost.ROSED$Is_Malicious)-1
test.SensorData.xgboost.ROSED$Is_Malicious = as.numeric(test.SensorData.xgboost.ROSED$Is_Malicious)-1

train.SensorData.xgboost.ROSED.DMatrix = xgboost::xgb.DMatrix(data = as.matrix(train.SensorData.xgboost.ROSED[, -which(names(train.SensorData.xgboost.ROSED) == "Is_Malicious")]), label = train.SensorData.xgboost.ROSED$Is_Malicious)

test.SensorData.xgboost.ROSED.DMatrix = xgboost::xgb.DMatrix(data = as.matrix(test.SensorData.xgboost.ROSED[, -which(names(test.SensorData.xgboost.ROSED) == "Is_Malicious")]))
set.seed(123)
# XGBoost parameters:
XGBoost.ROSED.params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta = 0.3,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 1,
  lambda = 1,
  alpha = 0.5
)

# Number of rounds for training:
nrounds <- 100

# Cross-validation to assess model performance:
XGBoost.ROSED.CVResults <- xgboost::xgb.cv(
  params = XGBoost.ROSED.params,
  data = train.SensorData.xgboost.ROSED.DMatrix,
  nrounds = nrounds,
  nfold = 5,
  metrics = "error",
  early_stopping_rounds = 15,
  showsd = TRUE,
  stratified = TRUE,
  print_every_n = 10
)

# Training xgboost model:
SensorData.xgboostModel.ROSED = xgboost::xgb.train(
  params = XGBoost.ROSED.params,
  data = train.SensorData.xgboost.ROSED.DMatrix,
  nrounds = XGBoost.ROSED.CVResults$best_iteration
)

# xgboost model predictions:
SensorData.xgboostModel.ROSED.predicted = predict(SensorData.xgboostModel.ROSED, test.SensorData.xgboost.ROSED.DMatrix)
SensorData.xgboostModel.ROSED.predictedLabels = ifelse(SensorData.xgboostModel.ROSED.predicted > 0.5, 1, 0)

SensorData.xgboostModel.ROSED.confM = table(Predicted = SensorData.xgboostModel.ROSED.predictedLabels, Actual = test.SensorData.xgboost.ROSED$Is_Malicious)
printMatrixInfoXGBoost(SensorData.xgboostModel.ROSED.confM, "XGBoost Model(Balanced data)")

########################################################################