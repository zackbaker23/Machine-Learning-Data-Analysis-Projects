#Load packages
library(tidyverse)
library(readxl)
library(glmnet)

### INITIAL ANALYSIS

#Read in the data
Final_Data <- read_excel("Final_Data.xlsx", na = "NA")

#I removed columns in the data with more than 50% missing after using other
#online sources to fill in the gaps

#There is certainly a high degree of multicollinearity here! 
#The lasso regression model can help with variable selection when there is 
#a high degree of multicollinearity between variables. Let's try it.

### Lasso regression model ###

#Predictor variables
x <- model.matrix(`Happiness Score`~., Final_Data)[,-1]

#Response variable
y <- Final_Data$`Happiness Score`

#Find the best lambda using cross-validation
set.seed(123)
cv <- cv.glmnet(x, y, alpha = 1)

# Display the best lambda value
cv$lambda.min

# Fit the model on the data
model_1 <- glmnet(x, y, alpha = 1, lambda = cv$lambda.min)

# Display regression coefficients
coef(model_1)

#Create a new dataset with only the candidate variables selected here
sub_data <- Final_Data %>% dplyr::select(`Happiness Score`,`GNI per capita, Atlas method (current US$)`,
                                         `Government expenditure on education, total (% of GDP)`,
                                         `Individuals using the Internet (% of population)`,
                                         `Life expectancy at birth, total (years)`,
                                         `Nurses and midwives (per 1,000 people)`,
                                         `PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)`,
                                         `Population growth (annual %)`, `Urban population (% of total population)`)
#I will use this data in SAS from here.