library(leaps)
library(glmnet)
library(splines)
library(FNN)
library(earth)
library(ggplot2)
library(ggthemes)
library(cluster)
library(fpc)

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

#Here is where I read in my .xlsx file, containing eventdata information
eventsdata <- read_excel("~/EventBrite Assignment/AssociateDataScientistHomework_DATA.xlsx")

# Here I convert the time data into numeric (for clustering analysis)
eventsdata$event_start_date = as.numeric(eventsdata$event_start_date)
eventsdata$event_end_date = as.numeric(eventsdata$event_end_date)


# Here I remove NA's from data
eventsdata = eventsdata[complete.cases(eventsdata),]


#I split my events data into 10 sets of training and testing data, in order to perform a 10-fold cross-
#validation later during my model assessment/selection.


cohort = split(eventsdata, eventsdata$category)
Business = cohort$`Business & Professional`
Other = cohort$`Other`
Welness = cohort$`Health & Wellness`
Music = cohort$Music
Family = cohort$`Family & Education`

kmeansdata = eventsdata[ -c(1,2,3,4,5,9)]
set.seed(3)
km.out = kmeans(kmeansdata,4,nstart=20)
km.out$centers


plotcluster(kmeansdata, km.out$cluster)

#This algorithm generates a random integer from 1 to 10 uniformly, and indexes it to the 
#training group dataframe, so that I can split into 10 different groups randomly 
#for 10-fold validation later in the analysis.
set.seed(2)
group = floor(runif(65534,1,11))
for (i in 1:65534)
{
  eventsdata$group[i] = group[i]
}


#Here I split the eventsdata into the 10
#different groups
groupsplit = split(eventsdata, eventsdata$group)
group1 =groupsplit$'1'
group2 =groupsplit$'2'
group3 =groupsplit$'3'
group4 =groupsplit$'4'
group5 =groupsplit$'5'
group6 =groupsplit$'6'
group7 =groupsplit$'7'
group8 =groupsplit$'8'
group9 =groupsplit$'9'
group10 =groupsplit$'10'


#combine groups to create train groups and test groups for 10-fold validation
train1 = rbind(group1,group2,group3,group4, group5, group6, group7, group8, group9)
test1 = group10
train2 = rbind(group1,group2,group3,group4, group5, group6, group7, group8, group10)
test2 = group9
train3 = rbind(group1,group2,group3,group4, group5, group6, group7, group9, group10)
test3 = group8
train4 = rbind(group1,group2,group3,group4, group5, group6, group8, group9, group10)
test4 = group7
train5 = rbind(group1,group2,group3,group5, group5, group7, group8, group9, group10)
test5 = group6
train6 = rbind(group1,group2,group3,group4, group6, group7, group8, group9, group10)
test6 = group5
train7 = rbind(group1,group2,group3,group5, group6, group7, group8, group9, group10)
test7 = group4
train8 = rbind(group1,group2,group4,group5, group6, group7, group8, group9, group10)
test8 = group3
train9 = rbind(group1,group3,group4,group5, group6, group7, group8, group9, group10)
test9 = group2
train10 = rbind(group2,group3,group4,group5, group6, group7, group8, group9, group10)
test10 = group1






#Used a series of models to see if 
#event_start_date, and event_end_date, and capaciy had an influence on the number of orders

lm.fit1 = lm(orders~event_start_date+event_end_date+capacity, data=train1)  
lm.fit2 = lm(orders~event_start_date+event_end_date+capacity, data=train2)  
lm.fit3 = lm(orders~event_start_date+event_end_date+capacity, data=train3)  
lm.fit4 = lm(orders~event_start_date+event_end_date+capacity, data=train4)  
lm.fit5 = lm(orders~event_start_date+event_end_date+capacity, data=train5)
lm.fit6 = lm(orders~event_start_date+event_end_date+capacity, data=train6)
lm.fit7 = lm(orders~event_start_date+event_end_date+capacity, data=train7)
lm.fit8 = lm(orders~event_start_date+event_end_date+capacity, data=train8)
lm.fit9 = lm(orders~event_start_date+event_end_date+capacity, data=train9)
lm.fit10 = lm(orders~event_start_date+event_end_date+capacity, data=train10)

#And these are my fitted values for each of the 10 linear models
predictlm.fit1 = predict(lm.fit1, data.frame(subset(test1,select=c(event_start_date,event_end_date,capacity))), interval="confidence")
predictlm.fit2 = predict(lm.fit2, data.frame(subset(test2,select=c(event_start_date,event_end_date,capacity))), interval="confidence")
predictlm.fit3 = predict(lm.fit3, data.frame(subset(test3,select=c(event_start_date,event_end_date,capacity))), interval="confidence")
predictlm.fit4 = predict(lm.fit4, data.frame(subset(test4,select=c(event_start_date,event_end_date,capacity))), interval="confidence")
predictlm.fit5 = predict(lm.fit5, data.frame(subset(test5,select=c(event_start_date,event_end_date,capacity))), interval="confidence")
predictlm.fit6 = predict(lm.fit6, data.frame(subset(test6,select=c(event_start_date,event_end_date,capacity))), interval="confidence")
predictlm.fit7 = predict(lm.fit7, data.frame(subset(test7,select=c(event_start_date,event_end_date,capacity))), interval="confidence")
predictlm.fit8 = predict(lm.fit8, data.frame(subset(test8,select=c(event_start_date,event_end_date,capacity))), interval="confidence")
predictlm.fit9 = predict(lm.fit9, data.frame(subset(test9,select=c(event_start_date,event_end_date,capacity))), interval="confidence")
predictlm.fit10 = predict(lm.fit10, data.frame(subset(test10,select=c(event_start_date,event_end_date,capacity))), interval="confidence")


#This is where I compute my L2 errors (Frebenius norm), and see how well my model performs
actualtest1 = test1$orders
fittedtest1 = predictlm.fit1[,1]
errorlm.fit1 = sqrt(sum((actualtest1-fittedtest1)^2))  
sqrt(sum((actualtest1-fittedtest1)^2))  

actualtest2 = test2$orders
fittedtest2 = predictlm.fit2[,1]
errorlm.fit2 = sqrt(sum((actualtest2-fittedtest2)^2))  
sqrt(sum((actualtest2-fittedtest2)^2))  

actualtest3 = test3$orders
fittedtest3 = predictlm.fit3[,1]
errorlm.fit3 = sqrt(sum((actualtest3-fittedtest3)^2))
sqrt(sum((actualtest3-fittedtest3)^2))

actualtest4 = test4$orders
fittedtest4 = predictlm.fit4[,1]
errorlm.fit4 = sqrt(sum((actualtest4-fittedtest4)^2))
sqrt(sum((actualtest4-fittedtest4)^2)) 

actualtest5 = test5$orders
fittedtest5 = predictlm.fit5[,1]
errorlm.fit5 = sqrt(sum((actualtest5-fittedtest5)^2))  
sqrt(sum((actualtest5-fittedtest5)^2)) 

actualtest6 = test6$orders
fittedtest6 = predictlm.fit6[,1]
errorlm.fit6 = sqrt(sum((actualtest6-fittedtest6)^2))  
sqrt(sum((actualtest6-fittedtest6)^2))  

actualtest7 = test7$orders
fittedtest7 = predictlm.fit7[,1]
errorlm.fit7 = sqrt(sum((actualtest7-fittedtest7)^2))  
sqrt(sum((actualtest7-fittedtest7)^2))  

actualtest8 = test8$orders
fittedtest8 = predictlm.fit8[,1]
errorlm.fit8 = sqrt(sum((actualtest8-fittedtest8)^2))
sqrt(sum((actualtest8-fittedtest8)^2))

actualtest9 = test9$orders
fittedtest9 = predictlm.fit9[,1]
errorlm.fit9 = sqrt(sum((actualtest9-fittedtest9)^2))
sqrt(sum((actualtest9-fittedtest9)^2)) 

actualtest10 = test10$orders
fittedtest10 = predictlm.fit10[,1]
errorlm.fit10 = sqrt(sum((actualtest10-fittedtest10)^2))  
sqrt(sum((actualtest10-fittedtest10)^2)) 

#model 1 test error is 32916.24
modelerror1 = mean(c(errorlm.fit1,errorlm.fit2,errorlm.fit3,errorlm.fit4,errorlm.fit5,errorlm.fit6,errorlm.fit7,errorlm.fit8,errorlm.fit9,errorlm.fit10))
mean(c(errorlm.fit1,errorlm.fit2,errorlm.fit3,errorlm.fit4,errorlm.fit5,errorlm.fit6,errorlm.fit7,errorlm.fit8,errorlm.fit9,errorlm.fit10))


## Below we use general-linear models (ridge regression)

## RIdge 1

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train1,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train1,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror1 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test1,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror1[i] = sqrt(sum((actualtest1-ridge.pred)^2))
  
}

min(ridgeerror1) 



##Ridge 2

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train2,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train2,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror2 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test2,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror2[i] = sqrt(sum((actualtest2-ridge.pred)^2))
  
}

min(ridgeerror2)




## Ridge 3

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train3,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train3,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror3 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test3,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror3[i] = sqrt(sum((actualtest3-ridge.pred)^2))
  
}

min(ridgeerror3) 



## Ridge 4

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train4,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train4,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror4 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test4,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror4[i] = sqrt(sum((actualtest4-ridge.pred)^2))
  
}

min(ridgeerror4) 




## Ridge 5

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train5,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train5,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror5 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test5,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror5[i] = sqrt(sum((actualtest5-ridge.pred)^2))
  
}

min(ridgeerror5) 

## Ridge 5

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train6,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train6,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror6 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test6,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror6[i] = sqrt(sum((actualtest6-ridge.pred)^2))
  
}

min(ridgeerror6)



## Ridge 7

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train7,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train7,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror7 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test7,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror7[i] = sqrt(sum((actualtest7-ridge.pred)^2))
  
}

min(ridgeerror7) 

## Ridge 8

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train8,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train8,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror8 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test8,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror8[i] = sqrt(sum((actualtest8-ridge.pred)^2))
  
}

min(ridgeerror8) 


## Ridge 9

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train9,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train9,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror9 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test9,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror9[i] = sqrt(sum((actualtest9-ridge.pred)^2))
  
}

min(ridgeerror9) 



## Ridge 10

grid = 10^seq(10,-2,length=100)
x1 = data.matrix(subset(train10,select=c(event_start_date,event_end_date,capacity)))
y1 = data.matrix(subset(train10,select=c(orders)))
ridge.mod = glmnet(x1,y1,alpha = 0, lambda=grid)
ridgeerror10 = rep(0,100)

for(i in 1:100)
{
  ridge.pred = predict(ridge.mod,s=grid[i], newx = data.matrix(subset(test10,select=c(event_start_date,event_end_date,capacity))))
  ridgeerror10[i] = sqrt(sum((actualtest10-ridge.pred)^2))
  
}

min(ridgeerror10) 

# Error of 32901.9, Better Model
mean(c(min(ridgeerror1),min(ridgeerror2),min(ridgeerror3),min(ridgeerror4),min(ridgeerror5),min(ridgeerror6),min(ridgeerror7),min(ridgeerror8),min(ridgeerror9),min(ridgeerror10)))


##Applying a Multivariate Adadptive Regression Spline

errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train1)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test1,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[1] = sqrt(sum((actualtest1-predict.spline)^2))

errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train2)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test2,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[2] = sqrt(sum((actualtest2-predict.spline)^2))


errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train3)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test3,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[3] = sqrt(sum((actualtest3-predict.spline)^2))


errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train4)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test4,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[4] = sqrt(sum((actualtest4-predict.spline)^2))


errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train5)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test5,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[5] = sqrt(sum((actualtest5-predict.spline)^2))


errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train6)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test6,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[6] = sqrt(sum((actualtest6-predict.spline)^2))


errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train7)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test7,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[7] = sqrt(sum((actualtest7-predict.spline)^2))

errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train8)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test8,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[8] = sqrt(sum((actualtest8-predict.spline)^2))

errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train9)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test9,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[9] = sqrt(sum((actualtest9-predict.spline)^2))

errorspline = rep(0,10)
splinemod = earth(orders~event_start_date+event_end_date+capacity, degree=3, pmethod = "forward", nfold=10, data=train10)
summary(splinemod)
predict.spline = predict(splinemod, data.frame(subset(test10,select=c(event_start_date,event_end_date, capacity))))
predict.spline
errorspline[10] = sqrt(sum((actualtest10-predict.spline)^2))



mean(errorspline)  ## Averager error for spline was 7870.91. By far the best model in this analysis.


#Here I create a plots to visualize my events category data
#Bar Plot on Category
a=ggplot(eventsdata,aes(x=category))
a+geom_bar(aes(fill=category),colour="orange")+
  ylab(label="Frequency")+
  xlab("Events")+
  theme(legend.position="right")+
  theme_economist()


eventsdata$timelength = eventsdata$event_end_date - eventsdata$event_start_date

#Here I create a line plot to visualize trend in event length (in time) and orders
#Bar Plot on Category
a=ggplot(eventsdata,aes(x=event_start_date, y=orders))
a+geom_jitter(colour="purple")+
  ylab(label="Frequency")+
  xlab("Events")+
  scale_x_continuous(limits = c(1382000000, 1518000000))+
  scale_y_continuous(limits = c(0, 350))
  #scale_fill_discrete(name="Gender",labels=c(1:20))+
  theme(legend.position="right")+
  theme_economist()

# Here I remove the outliers for the order dataset and look at the distribution of frequency of Number of Orders
a=ggplot(eventsdata,aes(x=orders))
a+geom_histogram(binwidth=6, colour="red", fill="yellow")+
  ylab(label="Frequency")+
  xlab("Number of Orders")+
  scale_fill_discrete(name="Dog",labels=c("predicted","actual"))+
  scale_x_continuous(limits = c(0, 67))+
  scale_y_continuous(limits = c(0, 15000))
  theme(legend.position="right")+
  theme_hc()

# Here I remove the outliers for the Capacity dataset and look at the Distribution of Capacity
  a=ggplot(eventsdata,aes(x=timelength))
  a+geom_histogram(binwidth=2500, colour="red", fill="blue")+
    ylab(label="Orders")+
    xlab("Length of time (eventend - eventstart)")+
    scale_fill_discrete(name="Dog",labels=c("predicted","actual"))+
    scale_x_continuous(limits = c(0, 34200))+
    scale_y_continuous(limits = c(0, 15000))+
  theme(legend.position="right")+
  theme_calc()
  


