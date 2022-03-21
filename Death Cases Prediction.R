# India Covid Cases 
# Time Series Prediction for Death Cases 


# Using Linear Regression 
# Set working directory 
setwd("D:/21_ Sem 6/Health Informatics/Project/Data")

# Load the Indian Dataset 
indian_cases <- read.csv('IndiaData.csv',  header=TRUE)
head(indian_cases)
dim(indian_cases)   # 457 * 59 

# Change the format of the date to Y-M-D
indian_cases$Date_reported<- format(as.Date(indian_cases$Date_reported, format = "%d/%m/%Y"), "%Y-%m-%d")

# Obtain only needed column 
indian_cases <- indian_cases[, c(4:6,8:9) ]
dim(indian_cases)   # 457 * 5 


# Add date and combine date and predictions into dataframe 
day_predicted <- c("2021-05-01","2021-05-02","2021-05-03","2021-05-04","2021-05-05",
                   "2021-05-06","2021-05-07","2021-05-08","2021-05-09","2021-05-10",
                   "2021-05-11","2021-05-12","2021-05-13","2021-05-14","2021-05-15",
                   "2021-05-16","2021-05-17","2021-05-18","2021-05-19","2021-05-20",
                   "2021-05-21","2021-05-22","2021-05-23","2021-05-24","2021-05-25",
                   "2021-05-26","2021-05-27","2021-05-28","2021-05-29","2021-05-30"
                   )
as.data.frame(day_predicted)



###################     LR      ########################################################
########################################################################################
# Predictions using linear regression  

date_ori <- indian_cases$Date_reported
# date_ori<- as.Date(date_ori)
# date_ori<- yday(date_ori)


# library(lubridate)
# indian_cases$Date_reported <- yday(indian_cases$Date_reported)

# Change the date to julian datatype
# Returns Julian day counts, date/time atoms from a 'timeDate' object
origin_date <- "2020-01-30"
origin_date <-as.Date(origin_date)
indian_cases$Date_reported <- as.Date(indian_cases$Date_reported)
indian_cases$Date_reported <- julian(indian_cases$Date_reported,origin=origin_date)

# Linear Regression Model 
set.seed(10)
lm_model <- lm(New_deaths~Date_reported,data=indian_cases)

print(summary(lm_model))

##########################################################################################
# Prediction for next 31  days (in May)
new_date <- data.frame(Date_reported = c(457:486))
# new_date <- data.frame(Date_reported = c(30:366))
prediction_result <-  predict(lm_model,new_date)


# Chart Plotting 
prediction_result<- as.numeric(prediction_result)
prediction_result<- as.data.frame(prediction_result)
whole <- cbind(day_predicted,prediction_result)
whole <- as.data.frame(whole)
class(whole)
dim(whole)


###########################    ARIMA     ############################################
#####################################################################################
# Using ARIMA as time-series model prediction  

# Using Linear Regression 
setwd("D:/21_ Sem 6/Health Informatics/Project/Data")

# Load the Indian Dataset 
indian_cases <- read.csv('IndiaData.csv',  header=TRUE)
head(indian_cases)
dim(indian_cases)   # 457 * 59 

# Change the format of the date to Y-M-D
indian_cases$Date_reported<- format(as.Date(indian_cases$Date_reported, format = "%d/%m/%Y"), "%Y-%m-%d")

indian_cases <- indian_cases[, c(4:6,8:9) ]
dim(indian_cases)   # 457 * 5 

# Make it sationary 
# install.packages("tseries")
library(tseries)
adf.test(diff(indian_cases$New_deaths), alternative = "stationary", k=0)

acf(indian_cases$New_deaths)
acf(diff(indian_cases$New_deaths))   # 0
pacf(diff(indian_cases$New_deaths))  # 5


# Fit the model (0,1,1)/ (2,2,3)
(fit <- arima(indian_cases$New_deaths, c(1, 1, 1),seasonal = list(order = c(1, 1, 1), period = 1)))

# Prediction the future 10 days
# pred <- predict(fit, indian_cases$New_deaths)
pred <- predict(fit, n.ahead = 30*1)
pred$pred 
prediction_result_arima <- as.data.frame(pred$pred)
prediction_result_arima <- as.numeric(pred$pred)
prediction_result_arima <- as.data.frame(pred$pred)


# Combine the prediction and date 
as.data.frame(day_predicted)
whole_arima <- cbind(day_predicted,prediction_result_arima)



#########################################################################################
##########################   LSTM   #####################################################
# Using LSTM 

library(keras)
library(tensorflow)
library(ggplot2)
library(timetk)

tensorflow::tf$random$set_seed(100)

# Using Linear Regression 
setwd("D:/21_ Sem 6/Health Informatics/Project/Data")

# Load the Indian Dataset 
indian_cases <- read.csv('IndiaData.csv',  header=TRUE)
head(indian_cases)
dim(indian_cases)   # 457 * 59 

# Change the format of the date to Y-M-D
indian_cases$Date_reported<- format(as.Date(indian_cases$Date_reported, format = "%d/%m/%Y"), "%Y-%m-%d")


indian_cases <- indian_cases[, c(4:6,8:9) ]
dim(indian_cases)   # 457 * 5 

# Rescale the input data 
scale_factors <- c(mean(indian_cases$New_deaths), sd(indian_cases$New_deaths))
scale_factors 

scaled_train <- indian_cases %>%
  dplyr::select(New_deaths) %>%
  dplyr::mutate(New_deaths = (New_deaths - scale_factors[1]) / scale_factors[2])
scaled_train 

# Lag for 31 
# Predict based on all the 10 past data
prediction <- 30
lag <- prediction

scaled_train <- as.matrix(scaled_train)

# we lag the data 19 times and arrange that into columns
x_train_data <- t(sapply(
  1:(length(scaled_train) - lag - prediction + 1),
  function(x) scaled_train[x:(x + lag - 1), 1]
))

# now we transform it into 3D form
x_train_arr <- array(
  data = as.numeric(unlist(x_train_data)),
  dim = c(
    nrow(x_train_data),
    lag,
    1
  )
)

x_train_arr

# Transform y into 3D form 
y_train_data <- t(sapply(
  (1 + lag):(length(scaled_train) - prediction + 1),
  function(x) scaled_train[x:(x + prediction - 1)]
))

y_train_arr <- array(
  data = as.numeric(unlist(y_train_data)),
  dim = c(
    nrow(y_train_data),
    prediction,
    1
  )
)
y_train_arr

x_test <- indian_cases$New_deaths[(nrow(scaled_train) - prediction + 1):nrow(scaled_train)]
x_test

# scale the data with same scaling factors as for training
x_test_scaled <- (x_test - scale_factors[1]) / scale_factors[2]

# this time our array just has one sample, as we intend to perform one 1-day prediction
x_pred_arr <- array(
  data = x_test_scaled,
  dim = c(
    1,
    lag,
    1
  )
)

x_pred_arr

# Model Building 
# Stateful = True : allows the LSTM to use longer context at training time while
#                   constraining the number of steps back for the gradient computation
lstm_model <- keras_model_sequential()

# Original 
lstm_model %>%
  layer_lstm(units = 512,                      # size of the layer   # 1,452,1 or 1,10,1
             batch_input_shape = c(1, 30, 1),  # batch size, timesteps, features
             return_sequences = TRUE,
             stateful = TRUE) %>%
 layer_dropout(rate = 0.5) %>%
 # layer_lstm(units = 512,
 #             return_sequences = TRUE,
 #            stateful = TRUE) %>%
 # layer_dropout(rate = 0.5) %>%
  time_distributed(keras::layer_dense(units = 1))


# Compile the model
lstm_model %>%
  compile(loss = 'mae', optimizer = 'adam',metrics = 'accuracy')

summary(lstm_model)

# shuffle = FALSE to preserve sequences of time series.

history <- lstm_model %>% fit(
  x = x_train_arr,
  y = y_train_arr,
  batch_size = 1,
  epochs = 8,
  verbose = 0,
  shuffle = FALSE    
)

plot(history)

lstm_forecast <- lstm_model %>%
  predict(x_pred_arr, batch_size = 1) %>%
  .[, , 1]

# we need to rescale the data to restore the original values
lstm_forecast <- lstm_forecast * scale_factors[2] + scale_factors[1]

fitted <- predict(lstm_model, x_train_arr, batch_size = 1) %>%
  .[, , 1]

lstm_forecast

# Forecast for one month in May 2021 (30 days)
lstm_forecast <- timetk::tk_ts(lstm_forecast,
                               start = c(457),
                               #end = c(2021, 10),
                               end = c(486),
                               frequency = 1
)
lstm_forecast

# convert time series prediction result to data frame 
prediction_lstm <- as.data.frame(lstm_forecast)
whole_lstm <- cbind(day_predicted,prediction_lstm)


######################   COMBINE PLOTTING   ####################################################
whole         # LM Prediction
whole_arima   # ARIMA Prediction
whole_lstm    # LSTM Prediction


library(ggplot2)
library(scales)

ggplot() +
  geom_line(data=indian_cases, aes(x=as.Date(Date_reported), y=New_deaths, group=1,color="Training"))+
  geom_point(data=indian_cases, aes(x=as.Date(Date_reported), y=New_deaths), color="blue", size=0.1) +
  geom_line(data=whole, aes(x=as.Date(day_predicted), y=prediction_result, group=2,color="Linear Regression Prediction"))+
  geom_point(data=whole, aes(x=as.Date(day_predicted), y=prediction_result), color="red", size=0.1) + 
  geom_line(data=whole_arima, aes(x=as.Date(day_predicted), y=x, group=1,color="ARIMA Prediction"))+
  geom_point(data=whole_arima, aes(x=as.Date(day_predicted), y=x), color="orange", size=0.1) +  
  geom_line(data=whole_lstm, aes(x=as.Date(day_predicted), y=x, group=1,color="LSTM Prediction"))+
  geom_point(data=whole_lstm, aes(x=as.Date(day_predicted), y=x), color="purple", size=0.1) + 
  labs(color="Legend", x = "Time Period",y = "Number of Death Cases",title ="Numbers of Daily Covid-19 Death Cases and Forecasting in India") +
  scale_x_date(breaks = date_breaks("months"),
               labels = label_date_short()) +
  scale_color_manual(values=c("orange","red","purple", "blue"))+
  scale_size_manual(values=c(3, 3))


#######################   Evaluating the Model     ##############################################################################
# Load the Indian Dataset 
may_cases <- read.csv('May Actual Cases.csv',  header=TRUE)
head(may_cases)
dim(may_cases)   # 20 * 59 

may_cases <- may_cases[, c(5:6,8:9) ]
dim(may_cases)   # 20 * 4 
may_cases <- cbind(day_predicted,may_cases)


# Calculate the RMSE between actual and predicted value 
library(Metrics)

# rmse(actual, predicted)
rmse_lm <- rmse(may_cases$new_deaths,whole[,2])
rmse_lm      #3096.088
rmse_arima <- rmse(may_cases$new_deaths,whole_arima[,2]) 
rmse_arima   #2482.497
rmse_lstm <- rmse(may_cases$new_deaths,whole_lstm[,2]) 
rmse_lstm    #2785.57


# Plot the actual and predicted 
ggplot() +
  geom_line(data=may_cases, aes(x=as.Date(day_predicted), y=new_deaths, group=1,color="Actual"))+
  geom_point(data=may_cases, aes(x=as.Date(day_predicted), y=new_deaths), color="blue", size=0.1) +
  geom_line(data=whole, aes(x=as.Date(day_predicted), y=prediction_result, group=2,color="Linear Regression Prediction"))+
  geom_point(data=whole, aes(x=as.Date(day_predicted), y=prediction_result), color="red", size=0.1) + 
  geom_line(data=whole_arima, aes(x=as.Date(day_predicted), y=x, group=1,color="ARIMA Prediction"))+
  geom_point(data=whole_arima, aes(x=as.Date(day_predicted), y=x), color="orange", size=0.1) +  
  geom_line(data=whole_lstm, aes(x=as.Date(day_predicted), y=x, group=1,color="LSTM Prediction"))+
  geom_point(data=whole_lstm, aes(x=as.Date(day_predicted), y=x), color="purple", size=0.1) + 
  labs(color="Legend", x = "Time Period",y = "Number of Death Cases",title =" Actual vs Predicted Numbers of Daily Covid-19 Death Cases in May 2021") +
  scale_color_manual(values=c("blue","orange","red", "purple"))+
  scale_size_manual(values=c(3, 3)) + 
  scale_x_date(breaks = date_breaks("days"),
               labels = label_date_short()) 


