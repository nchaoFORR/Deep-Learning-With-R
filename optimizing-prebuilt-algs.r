library(keras)

# we'll be using the VGG16 algorithm, a cnn for the ImageNet.

conv_base <- application_vgg16(
  
  # weight checkpoint to initialize model from
  weights = "imagenet",
  # include densely connected classifier it comes with
  # it comes from imagenet, which has 1000 classes. We only want 2, so we'll build our own classifer
  include_top = FALSE,
  # if you omit input shape, the net can take input of any size
  input_shape = c(150, 150, 3)
  
)


###

# 2 ways of approaching this:

# 1. Run images through the convolutional layers and save output to the disk. Then train a separate
#    densely connected classifier on that output. Computationally cheap, no data augmentation.

# 2. Build a densely connected classifier on top of this net. More flexibility and data augmentation, computationally expensive.

## Approach 1:

base_dir <- file.path("~/Documents/deep-learning-with-r/dogs-vs-cats/cats-and-dogs-small")

train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")


data_gen <- image_data_generator(rescale = 1/255)

batch_size <- 20

# build function that will train on a sample from the directory in batches
extract_features <- function(directory, sample_size) {
  
  # function will fill up these features and labels in batches (looping through as many batches until you gone through all images)
  features <- array(0, dim = c(sample_size, 4, 4, 512))
  
  labels <- array(0, dim = sample_size)
  
  # data generator (from directory)
  generator <- flow_images_from_directory(
    
    directory = directory,
    generator = data_gen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
    
  )
  
  i <- 0
  # this will just be an indefinite loop that we'll cut off after going through an epoch
  while(TRUE) {
    
    # grab batch
    batch <- generator_next(generator)
    
    # separate inputs and labels
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    # predict with pre-built model on inputs_batch to create input features of our final model
    features_batch <- 
      conv_base %>% 
      predict(inputs_batch)
    
    # create index range 1:20, 21:40, 41:60, etc.
    index_range <- ((i * batch_size) + 1):((i + 1) * batch_size)
    
    # get out input features for the classifier (the output of vgg16)
    features[index_range, , , ] <- features_batch
    # get our labels to train with
    labels[index_range] <- labels_batch
    
    # repeat or next batch
    i <- i + 1
    
    # break after going through the data
    if(i * batch_size >= sample_size) {
      
      break
      
    }
    
  }
  
  # final_output
  list(
    
    features = features,
    labels = labels
    
  )
  
}

# lets create our train validation and test data for the classifer
train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)

# and test
test <- extract_features(test_dir, 1000)

# densely connected layer needs to take a 2d tensor as input
# currently, our feature data is shape (sample_size, 4, 4, 512)
# we need to flatten to shape (sample_size, 4 * 4 * 512) == (sample_size, 8192)

# build reshape function
reshape_features <- function(array) {
  
  array_reshape(array, dim = c(nrow(array), 8192))
  
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)


# create out densely connected classifer, train it and predict.
# we'll use dropout for regularization

model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(8192)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% 
  compile(
    
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
    
  )


# train
history <- model %>% 
  fit(
    
    train$features,
    train$labels,
    
    epochs = 30,
    batch_size = 20,
    
    validation_data = list(validation$features, validation$label)
    
  )


# viz
plot(history)



# great model at 90+ % accuracy, however we do still see some overfitting

#######################
# Approach 2:

# we can now use data augmentation to reduce overfitting

# we can just go ahead and build our model
model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model

# note -- ~17 million parameters. this is a computationally expensive task!

# FREEZE CONV BASE BEFORE TRAINING!!!
#   our new densly connected classifier will start with random weights, and the first few
#   optimization steps will be large. these large changes (back)propagate through the network
#   and will destroy our pre-train base. we must FREEZE the base.

freeze_weights(conv_base)

# train this guy

# augment data
train_datagen <- image_data_generator(
  
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = T,
  fill_mode = "nearest"
  
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
  
)

validation_generator <- flow_images_from_directory(
  
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
  
)

model %>% 
  compile(
    
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 2e-5),
    metrics = c("accuracy")
    
  )

history <- model %>% 
  fit_generator(train_generator,
                steps_per_epoch = 100,
                epochs = 30,
                validation_data = validation_generator,
                validation_steps = 50)


plot(history)

# netter final accuracy score >90% and overfitting is no longer occurring.



### Second optimization option: FINE TUNING

# in fine-tuning, we take the final few layers of the pre-trained convultional layers and unfreeze them
# during training. This allows our model to re-train itself on the most abstract features
# of the original images.

# we run into the same dilemma as before; randomly initializing our final dense layers will ruin
# our convolutional base whn its unfrozen. For this reason, we:
#   1. Free base and train dense layers.
#   2. unfreeze base and retrain some of the final convolutional layers along with dense layers.

# we already did the first step, so lets apply step 2 to it.

# unfreeze some later convolutional layers (block 3 and onwards)
unfreeze_weights(conv_base, from = "block3_conv1")

# retrain/fine-tune
model %>% 
  compile(
    
    loss = "binary_crossentropy",
    # we will use a very low learning rate.
    # this is to prevent major adjustments to be made to
    # the fine-tune layers. large adjustments will mess up
    # the abstract representations they are previously tuned to.
    optimizer = optimizer_rmsprop(lr = 1e-5),
    metrics = c("accuracy")
    
  )

history <- model %>% 
  fit_generator(
    
    train_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50
    
  )

plot(history)

model %>% save_model_hdf5("cat-dog-small-fine-tune.h5")

# predict
test_generator <- flow_images_from_directory(
  
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
  
)

model %>% evaluate_generator(
  
  test_generator,
  steps = 50
  
)

# 97% accuracy