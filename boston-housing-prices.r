library(keras)
library(dplyr)
library(gpplot2)

dataset <- dataset_boston_housing()

c(c(train_x, train_y), c(test_x, test_y)) %<-% dataset

# rescale our data
# remember, use train parameters to scale test data.
# this ensures that nothing from the test set ever enters our workflow.

mean_train <- apply(train_x, 2, mean)
sd_train <- apply(train_x, 2, sd)

train_x <- scale(train_x, center = mean_train, scale = sd_train)
test_x <- scale(test_x, center = mean_train, scale = sd_train)

# functionalize our model for cv
build_model <- function(){
  
  model <- keras_model_sequential() %>% 
    # remember, our input is a single observation. our input shape is thus 13
    layer_dense(units = 64, activation = "relu", input_shape = c(13)) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1)
  
  model %>% compile(
    
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c('mae')
    
  )
  
}

# cv to tune model
k <- 4

indices <- sample(1:nrow(train_x))
folds <- cut(indices, breaks = k, labels = FALSE)

num_epochs <- 100
all_mae_histories <- NULL
for(i in 1:k){
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  val_x <- train_x[val_indices, ]
  val_y <- train_y[val_indices]
  
  partial_x <- train_x[-val_indices, ]
  partial_y <- train_y[-val_indices]
  
  model <- build_model()
  
  history <- model %>% 
    fit(
      partial_x, partial_y,
      epochs = num_epochs,
      batch_size = 1,
      validation_data = list(val_x, val_y),
      verbose = 1
    )
  
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories <- rbind(all_mae_histories, mae_history)
  
}



average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

average_mae_history %>% 
  ggplot(aes(epoch, validation_mae)) +
  geom_smooth

model <- build_model()
model %>% 
  fit(train_x, train_y,
      epochs = 80, batch_size = 16,
      verbose = 1)

result <- model %>% evaluate(test_x, test_y)

result
