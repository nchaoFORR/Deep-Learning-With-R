# Machine Learning Fundamentals

### Evaluating Models

## Validation/Testing Methods:

# Simple Hold-out Validation

# worth getting used to splitting data the way its done in the book. Very concise and readable.

# df is a training data that needs to be split to eval data and train data

indices <- sample(1:nrow(df), size = 0.8 * nrow(df))

evaluation_data <- df[-indices, ]
training_data <- df[indices, ]

model <- get_model()

model %>% train(training_data)
validation_score <- model %>% evaluate(validation_data)

# before testing, it is common to retrain model using all data, test and training
model <- get_model()

model %>% train(df)

test_score <- model %>% evaluate(test_data)

# main problem: you need a lot of data for this, otherwise your evaluation data will be
# prone to random variance.

## K-Fold Validation

# split data into k parts. train the data on data[-k, ], then validation on data[k, ].
# do this for each partition. model score is average score of each training.

k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = 4, labels = FALSE)

validation_score <- c()

for(i in 1:k){
  
  validation_indices <- which(folds == i, arr.ind = TRUE)
  validation_data <- train_data[validation_indices, ]
  trainingi_data <- train_data[-validation_indices, ]
  
  model <- get_model()
  model %>% train(
    training_data
  )
  results <- model %>% evaluate(validation_data)
  validation_score <- c(validation_score, results$metric)
}

validation_score <- mean(validation_scores)

model <- get_model()
model %>% train(train_data)
results <- model %>% evaluation(test_data)

# great for relatively small datasets.
# main drawback: computationally expensive



### Data preprocessing and feature engineering

## most feature engineering is domain-specific (e.g. test data feature engineering)
# for now, we'll just do the universals

# vectorization, normalization, missing data, and feature extraction

## vectorization:
# neural nets must be fed tensors (of floating point data or integers). this is what
# vectorization accomplishes

# normalization:
# following attributes should be given to the data
# 1. small values -- values larger than the initial weight values is bad
# 2. be homogenous -- all values should be in roughly the same range

# following aren't always necesarry but still important to know
# normalize each feature independently to have mean 0
# normalize each feature independently to have sd 1

# remember when we divded the image data by 255? that was to force every value to
# be between 0 and 1 (instead of 0 and 255 which is greyscale).

# formally, normalizing data means giving it a normal distribution. (think about the grayscale)

# check out caret and recipes packages for preprocessing and normlaization

## missing data:
# unlike in traditional modeling, neural nets can learn to recognize missing values and ignore them.
# because of this, setting to 0 is generally safe unless 0 already has a meaningful value.
# NOTE: if your train data has missing values, it won't learn missing values! so if you know
# your test data will have missing values, you should artifically insert missing data

## feature-engineering
# could probably write a whole novel and then some...
# but worth exemplifying: imagine training a model to read time from a clock.
#   you could feed it images of the clock, or feature engineer vectors to represent the clock
#   hand positions. The latter will be the easier model to train by far!

### Regularization: its really a technique to fight against overfitting (makes sense when you think about lassos and ridge regressions)
## goal is to force the model to focus on the most prominent patterns
## techinques:

# - reduce network size; less learnable parameters (aka capacity) means less risk of overfitting (can reduce both number of layers and width of layers)
#    ---- THE CHALLENGE OF DEEP LEARNING IS GENERALIZING, NOT FITTING! A HUGE NEURAL NET CAN MEMORIZE A DICTIONARY!!!

# general workflow is to start with few layers and build out until you start seeing diminishing returns.

 # - weight regularization
#     tied to occam's razor -- given two models that perform similarly, opt for the simpler one. its less likely to overfit
#     - we force weights to be small by adding a penalty to our cost function due to high weights (same as regularized regression)
#     - smaller weights are more regular weights
#     - like with regressions, there are L1 models (absolute value of weights) and L2 models (square of weights, aka weight decay)
#     - also like regressions, there are elastic nets

#syntax:
layer_dense(units = 16, kernel_regularizer = regularizer_l1(0.002), # penalty term of 0.002*weight_coeff added to loss function
            activation = "relu", input_shape = c(10000))

# NOTE: This penalty term is only added when training the model. the test data should thus see a lower loss score.

# - adding dropout is another regularization technique
#      - each layer's output will have randomly chosen (but set number so next layer knows its input size)
#        weights set to 0. These weights are "dropped out" in that training sample. num of weights dropped is the dropout-rate (usually between 0.2 and 0.5)
#      - when evaluating on test data, no weights are dropped but all are scaled back by the dropout rate ( to balance the fact that now more wiehgts are active in test time).

# 50% dropout:
dropout_layer_output <- layer_output * sample(0:1. length(layer_output), replace = T)

dropout_test_layer_output <- test_layer_output * 0.5 # dropout-rate

# conceptually, we are adding noise to output values to break up random patterns that the model might be picking up on.

# syntax:
layer_dropout(rate = 0.5) %>% 
  layer_dense(...)



