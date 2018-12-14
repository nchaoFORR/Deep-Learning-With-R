# Chapter 5: Convolutional Neural Nets

library(keras)

mnist <- dataset_mnist()

# architecture of typical covnet:
model <- 
  keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

model

# as you go deeper into the neural net, the output shapes shrink.
# they do however stay as 3D tensors, while we need vectors for our classification
# layer. because of this, we will now pack this into a denseky connected classifier network.

model <- model %>%
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

model

# layer_flatten condensed into vectors of length 576 (vector-wise multiplication of the previous layer)

# lets train this model
c(c(train_x, train_y), c(test_x, test_y)) %<-% mnist

train_x <- array_reshape(train_x, c(60000, 28, 28, 1))
train_x <- train_x / 255

test_x <- array(test_x, c(10000, 28, 28, 1))
test_x <- test_x / 255

train_y <- to_categorical(train_y)
test_y <- to_categorical(test_y)


model %>% compile(
  
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
  
)

model %>% fit(
  
  train_x, train_y,
  epochs = 5,
  batch_size = 64
  
)


results <- model %>% evaluate(test_x, test_y)

results


# 99.1% accuracy.

# convnets look for local structures (of an image), rather than
# global structures like densely connected layers.
# the kernels are these local areas.

# 2 key characteristics:
# patterns detected are at local level, and so learned patterns can be 
# detected in any area of the space.
# they learn spatial hierarchies of patterns/structures, so they start by recognizing edges,
# the recognizing patterns in the output of that layer, and so on.


# how do convolutional layers work?

# a layer takes in a tensor input of dimensions c(width, height, depth).
#   - in our case, depth was a grayscale of depth 1 (i.e. gray-ness),
#   - while color images would have a depth of 3 (RGB).
# the layer takes many filters (kernels) that are much smaller than the image.
# those filters are mapped across the entire image. at each place, it transforms
# that area of the image into a vector of length num_filters.
# each area of the input image is then reassembled with the layer outputs.
# output has dimensions (~width, ~height, num_filters).
# width and height are approximates due to borders and padding.

