# trying blending models on titanic dataset

setwd("~/titanic")
train.data <- read.csv("train.csv", header=TRUE, stringsAsFactors=FALSE, 
                       na.strings =c("NA",""," "))
test.data <- read.csv("test.csv", header=TRUE, stringsAsFactors=FALSE, na.strings =c("NA",""," "))

library(dplyr)
library(Hmisc)
library(magrittr)
library(randomForest)
library(gbm)
library(caret)
library(randomForest)
library(pROC)
# combining both train and test data since both of them have missing data and we want to impute altogether
train.data <- mutate(train.data, Is.Train= TRUE)
test.data <- mutate(test.data, Is.Train= FALSE,Survived = NA)
complete.data <- rbind(train.data,test.data)
###########
# to impute missing data

#Now we extract the title from the names to extract characteristics and see if that has an impact. 
complete.data$Title <- gsub(".*\\,\\s(\\w*)\\..*","\\1",complete.data$Name)
title.vec <-(unique(complete.data$title))

# function that gets the mean for specific titles
get.mean <- function(titul){
  age.vec <- complete.data %>% filter(.,title==titul) %>% select(.,Age)
  age.mean <- median(age.vec$Age, na.rm=TRUE) %>% round(.,digits=1)
  return(age.mean)
}
# generates list for age for all titles
age.list <- sapply(title.vec, FUN=get.mean, USE.NAMES=TRUE)
# impute missing age values by average of the category
impute.age <- function(index){
  if(is.na(complete.data[index,]$Age)){
    ms.age <- age.list[[complete.data[index,]$title]]
  }
  else{
    ms.age <- complete.data[index,]$Age
  }
  return(ms.age)
}
complete.data$New.Age <- sapply(seq(1,nrow(complete.data), by =1), FUN=impute.age)
# inspect the data and drop seemingly unnecessary columns
complete.data <- subset(complete.data, select = -c(Name,Ticket,Cabin,Age))
str(complete.data)
# how many rows are there with NA now? other than survived of course

complete.data[!complete.cases(subset(complete.data, select =-c(Survived))),]
# small number, lets impute,
# to remove NA in embarked, we replace <NA> values with most frequent value for 1st class passengers S
filter(complete.data,Pclass==1)$Embarked %>% table(.)
complete.data[!complete.cases(complete.data),]$Embarked <- c("S","S")
# impute NA in fare, median of PClass=3 fare
m.fare <-complete.data %>% filter(.,Pclass==3) %>% select(.,Fare) 
# impute by hand code
complete.data[1044,]$Fare <- median(m.fare$Fare, na.rm=TRUE)
# lets inspect the data again
str(complete.data)
# lets factorize necessary columns

complete.data$Pclass <- as.factor(complete.data$Pclass)
complete.data$Embarked <- as.factor(complete.data$Embarked)
complete.data$Title <- as.factor(complete.data$Title)
complete.data$Sex <- as.factor(complete.data$Sex)
# now that all processing is done lets revert back to original training and test data set

train.data <- complete.data %>% filter(.,Is.Train == TRUE) %>% subset(.,select = -c(Is.Train))
test.data <- complete.data %>% filter(.,Is.Train == FALSE) %>% subset(.,select = -c(Is.Train, Survived))
# lets clear up the workspace 
rm(complete.data,m.fare, age.list, title.vec)
# lets have benchmark randomforest model
set.seed(11)
rf.model <- randomForest(x = subset(train.data, select = -c(Survived,PassengerId)), y = as.factor(train.data$Survived))
rf.model
?predict.randomForest
set.seed(14)
rf.prediction <- predict(rf.model, newdata=subset(test.data, select = -c(PassengerId)) )
# writing for kaggle submission
predict.df.benchmark <- data.frame(passengerID = test.data$PassengerId, Survived =rf.prediction)
write.csv(predict.df.benchmark,"bm.csv",row.names=FALSE)

# Kaggle score with RF model 0.79426 - this is the benchmark. We will see if we can improve upon it by blending

# now for ensemble blending we need to divide the training data between ensemble data and blending data and then blend by cross-validation

en.index <- createDataPartition(train.data$PassengerId, p=0.70, list=FALSE)[,1]
ensemble.data <- train.data[en.index,]
blend.data <- train.data[-en.index,]

# gbm model on ensemble data
gbm.model <- gbm.fit(x= subset(ensemble.data, select = -c(Survived,PassengerId)), y = (ensemble.data$Survived), distribution = "bernoulli",interaction.depth = 4,shrinkage = 0.001,n.trees = 10000)

# randomforest model on ensemble data
randf.model <- randomForest(x = subset(ensemble.data, select = -c(Survived,PassengerId)), y = as.factor(ensemble.data$Survived))

# now that we have two different models, lets predict for blended data

bl.pred.gbm <- predict(gbm.model, newdata = subset(blend.data, select = -c(Survived,PassengerId)), n.trees=10000)
bl.pred.rf <- predict(randf.model, newdata = subset(blend.data, select = -c(Survived,PassengerId)),type= "response")

# define a function to return classification values for gbm prediction
classification.function <- function(x){
  if(x<0.5){
    return(0)
  }
  else{
    return(1)
  }
}

blend.data <- mutate(blend.data,rf.predict = as.numeric(levels(bl.pred.rf))[bl.pred.rf] )
blend.data <- mutate(blend.data,gbm.predict = sapply(bl.pred.gbm, FUN=classification.function ) )

table(blend.data$rf.predict)
table(blend.data$gbm.predict)

# now lets try blended model with gbm combining RF and GBM

blend.gbm <- gbm.fit(x= subset(blend.data, select = c(gbm.predict,rf.predict)), 
                     y =  blend.data$Survived, n.trees=10000, shrinkage = 0.001)


blend.rf <- randomForest(x= subset(blend.data, select = c(gbm.predict,rf.predict)), 
                     y =  as.factor(blend.data$Survived))
###################################################

# now lets predict for test data in the same fashion

te.pred.gbm <- predict(gbm.model, newdata = subset(test.data, select = -c(PassengerId)), n.trees=10000)
te.pred.rf <- predict(randf.model, newdata = subset(test.data, select = -c(PassengerId)),type= "response")



test.data <- mutate(test.data,rf.predict = as.numeric(levels(te.pred.rf))[te.pred.rf] )
test.data <- mutate(test.data,gbm.predict = sapply(te.pred.gbm, FUN=classification.function ) )

# final step prediction with blended model

test.survival <- predict(blend.gbm, newdata = subset(test.data, select = c(gbm.predict,rf.predict)), n.trees= 10000)

## trying RF
test.survival <- predict(blend.rf, newdata = subset(test.data, select = c(gbm.predict,rf.predict)))

predict.survival <- sapply(test.survival, FUN=classification.function )

table(test.survival)

# writing to CSV file for Kaggle submission
predict.df.bl <- data.frame(passengerID = test.data$PassengerId, Survived =predict.survival)
write.csv(predict.df.bl,"bl.csv",row.names=FALSE)

# submission to kaggle gives 0.77512 not better than the benchmark model. This is not unexpected as both the models are very similar.
