#Load required packages
library(readr)
library(alr4)
library(dplyr)
library(ggplot2)
library(olsrr)

#Read the data file into nba_data
nba_data <- read_csv("NBA Data.csv")
View(nba_data)

################## DATA MANIPULATION #####################################

#Make followers and turnovers strictly positive. 
#This is necessary for the Box-Cox transformation we will perform later.
nba_data$followers <- nba_data$followers+0.01
nba_data$tov <- nba_data$tov+0.001

################ NEED FOR PREDICTOR TRANSFORMATION??? ##########################

#Variable scatterplot - followers, ftar clearly need transformation.
pairs(~ salary + age + netrtg + astpct + rebpct +
        usg + followers + tov + ftar + ws, data = nba_data)

#Multivariate Box-Cox tranformation of predictors
summary(trp <- powerTransform(cbind(age,astpct,rebpct,followers,usg,ftar,ws,tov)~1, data=nba_data))

############################# MODEL 1 ##########################################

#Initial model with predictors transformed according to MBC
#Contains all potential predictors.
init_model_1 <- lm(salary~pos+log(age)+netrtg+log(astpct)+rebpct+
                   usg+log(followers)+tov+I(ftar^(-1/2))+I(ws^(1/2)), data=nba_data)
summary(init_model_1)

#MODEL 1: Stepwise regression with transformed predictors
#and untransformed response. Only want to transform the 
#response variable if diagnostics look bad.
step(init_model_1, scope=~1, direction="backward")

summary(m1 <- lm(salary ~ log(age) + usg + log(followers) + tov + 
                    I(ws^(1/2)), data = nba_data))

################## RESIDUAL DIAGNOSTICS FOR MODEL 1 #############################

#Plot residual vs. fitted values
ols_plot_resid_fit(m1)

#Normal qqplot of residuals
ols_plot_resid_qq(m1)

#Residual histogram
ols_plot_resid_hist(m1)

################ NEED FOR RESPONSE TRANSFORMATION??? ##########################

#Transformation of response?
inverseResponsePlot(init_model_1)
boxCox(init_model_1)
summary(powerTransform(init_model_1))

############################ MODEL 2 ##########################################

#Model with transformed predictors and Salary^(1/2)
init_model_2 <- lm(I(salary^(1/2))~pos+log(age)+netrtg+log(astpct)+rebpct+
                     usg+log(followers)+tov+I(ftar^(-1/2))+I(ws^(1/2)), data=nba_data)
summary(init_model_2)

#Backward selection algorithm for new model
step(init_model_2, scope=~1, direction="backward")

#MODEL 2: Model with transformed predictors and response
m2 <- lm(I(salary^(1/2)) ~ log(age) + netrtg + usg + log(followers) + 
        tov + I(ws^(1/2)), data = nba_data)
summary(m2)

######################### RESIDUAL DIAGNOSTICS FOR M2 ###########################

#Plot residual vs. fitted values
ols_plot_resid_fit(m2)

#Normal qqplot of residuals
ols_plot_resid_qq(m2)

#Residual histogram
ols_plot_resid_hist(m2)

#Cook's distance plot
ols_plot_cooksd_bar(m2)

#Diffs plot
ols_plot_dffits(m2)
