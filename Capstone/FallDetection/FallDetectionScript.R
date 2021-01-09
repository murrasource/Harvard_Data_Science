# IMPORT LIBRARIES
list.of.packages <- c("tidyverse","dplyr","caret","lubridate",
"randomForest","rpart","rpart.plot","ggplot2","ggpubr","knitr","tinytex",
"car","gridExtra")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)>0){install.packages(new.packages)}
options(scipen = 999)

# IMPORT DATA
download.file("https://github.com/murrasource/Harvard_Data_Science/blob/main/Capstone/FallDetection/falldetection.csv", 
    destfile = "/falldetection.csv", method = "curl")
raw = read_csv("falldetection.csv")

# CLEAN DATA

#Friendly levels
falls = raw %>% mutate(
  Activity = recode(factor(ACTIVITY), 
                    recodes = "'0'='Standing';
                              '1'='Walking';
                              '2'='Sitting';
                              '3'='Falling';
                              '4'='Cramps';
                              '5'='Running'"),
  Time = TIME,
  SugarLevel = SL,
  EEG = EEG,
  BloodPressure = BP,
  HeartRate = HR,
  Circulation = CIRCLUATION
) %>% 
  select(Activity, Time, SugarLevel, EEG, BloodPressure, HeartRate, Circulation)

# Make binary Fall column
falls = falls %>% mutate(Fall = ifelse(Activity == "Falling", TRUE, FALSE))

# Filter outliers
falls = falls %>% filter(
  EEG > quantile(EEG, 0.25) - 1.5*IQR(EEG) & 
    EEG < quantile(EEG, 0.75) + 1.5*IQR(EEG) &
    SugarLevel > quantile(SugarLevel, 0.25) - 1.5*IQR(SugarLevel) &
    SugarLevel < quantile(SugarLevel, 0.75) + 1.5*IQR(SugarLevel) &
    Circulation > quantile(Circulation, 0.25) - 1.5*IQR(Circulation) &
    Circulation < quantile(Circulation, 0.75) + 1.5*IQR(Circulation) &
    BloodPressure > quantile(BloodPressure, 0.25) - 1.5*IQR(BloodPressure) &
    BloodPressure < quantile(BloodPressure, 0.75) + 1.5*IQR(BloodPressure) &
    HeartRate > quantile(HeartRate, 0.25) - 1.5*IQR(HeartRate) &
    HeartRate < quantile(HeartRate, 0.75) + 1.5*IQR(HeartRate)
)

# Train and Test sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(falls$Fall, times = 1, p = 0.1, list = FALSE)
train = falls[-test_index,]
test = falls[test_index,]

train = train %>% arrange(Time)
test = test %>% arrange(Time)

# Get rid of activity column
train = train %>% select(-Activity)
test = test %>% select(-Activity)

# train random forest
set.seed(2021, sample.kind = "Rounding")
train_control = trainControl(method="cv", number=10, savePredictions = TRUE)
rf_fit = train(
  factor(Fall) ~ .,
  data = train,
  method = "rf",
  trControl = train_control
  )

# create index of mistakes
row_index = rf_fit$pred %>% filter(pred != obs) %>% select(rowIndex) %>% as.matrix()

#####Create supplementary normalized data#####
# Using medians as the means for rnorm() to resist outliers
means = train[row_index,] %>% filter(Fall == TRUE) %>% summarize(Time = median(Time), SugarLevel = median(SugarLevel), EEG = median(EEG), BloodPressure = median(BloodPressure), HeartRate = median(HeartRate), Circulation = median(Circulation))
iqrs = train[row_index,] %>% filter(Fall == TRUE) %>% summarize(Time = IQR(Time), SugarLevel = IQR(SugarLevel), EEG = IQR(EEG), BloodPressure = IQR(BloodPressure), HeartRate = IQR(HeartRate), Circulation = IQR(Circulation))

# SDs are calculated from the IQR to ensure the SDs are not biased from outliers
sds = iqrs * 0.35

new_data = tibble(Time = vector(length = 100), SugarLevel = vector(length = 100), EEG = vector(length = 100), BloodPressure = vector(length = 100), HeartRate = vector(length = 100), Circulation = vector(length = 100))

for(feature in colnames(means)){
  mean = means[[feature]] %>% as.numeric()
  sd = sds[[feature]] %>% as.numeric()
  data = rnorm(100, mean = mean, sd = sd)
  new_data[[feature]] = data
}

new_data$Fall = rep(TRUE, 100)

added = rbind(train, new_data)
###############

# train RF on supplementary data
exp_fit = train(
  factor(Fall) ~ .,
  data = added,
  method = "rf",
  trControl = train_control
)

# train kNN on original train data

knn_fit = train(
  factor(Fall) ~ .,
  data = train,
  model = "knn",
  trControl = train_control
)


#### Ensemble predictions on test set
knn_predict = predict(knn_fit, test)
exp_predict = predict(exp_fit, test)
rf_predict = predict(rf_fit, test)


# COMBINED is the Guess vector
combined = tibble(knn = as.logical(knn_predict), 
                  exp = as.logical(exp_predict), 
                  rf = as.logical(rf_predict))

confusionMatrix(as.factor(ifelse(rowSums(combined) > 1, TRUE, FALSE)), 
                reference = as.factor(test$Fall),
                positive = "TRUE")


