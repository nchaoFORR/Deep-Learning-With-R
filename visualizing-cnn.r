library(keras)

# 3 key ways visualizing can assist in building cnn

# 1. visualize intermediate layers outputs (intermediate activations)
# 2. visualize the filter
# 3. visualize class activation heatmap (which region of an image was classified as what?)

dir_main <- file.path("~/Documents/deep-learning-with-r/")

# 1. Visualizing intermediate activations

# load the fully custom model we built earlier
model <- load_model_hdf5(file.path(dir_main, "dogs-vs-cats/cats-and-dogs_small_2.h5"))

model


# get some input image that we haven't trained the model on
img_path <- file.path(dir_main, "dogs-vs-cats/cats-and-dogs-small/test/cats/cat.1700.jpg")

# load image then process it into a 4-d vector
img <- image_load(img_path, target_size = c(150, 150))

# convert to array
img_tensor <- image_to_array(img) %>% 
  # reshape into 4d tensor and divide by 255 (this is how our training data was processed)
  array_reshape(c(1, 150, 150, 3)) / 255

dim(img_tensor)

# visualize
plot(as.raster(img_tensor[1, , , ]))



# to do this; 
# as we train the neural net, we want to output the activations from each convolutional and pooling layer.
# we will use keras_model() instead of keras_sequential_model() so that we can have multiple outputs

# first get the outputs of the first 8 layers of the model
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)

# instantiate our model with an input tensor and list of output tensors
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

# when fed an image, this model will output the activations from each of the first 8 layers of the original model
activations <- activation_model %>% 
  predict(img_tensor)

# lets check out the output of the first convulation
first_layer_activation <- activations[[1]] 
dim(first_layer_activation)

# 148 by 148 by 32
# to make things easier, we will want to visualize one channel (total 32) at a time as a 2d image.

plot_channel <- function(channel) {
  
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, col = terrain.colors(12))
  
}

# second channel -- catching on edges?
plot_channel(first_layer_activation[1, , , 2])

# seventh layer? -- catching on ears and eyes
plot_channel(first_layer_activation[1, , , 7])



# visualizing every channel

image_size <- 58
images_per_row <- 16

for(i in 1:8) {
  
  # get the i'th output for our input image
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0("cat_activations_", i, "_", layer_name, ".png"),
      width = image_size * images_per_row,
      height = image_size * n_cols)
  
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for(col in 0:(n_cols-1)) {
    for(row in 0:(images_per_row-1)) {
      
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
      print("message")
      
    }
  }
  
  par(op)
  dev.off()
  
}



# 2: Visualizing Filters
# done by applying gradient ascent in input space;
#  - starting frmo blank image, gradient descent to the value of the input image of a covnet, maximing the response of a specific filter
#   this will reveal an input image that the filter has a mximum response to

#  - build loss fromfunction that maximizes value of a filter
#  - use stochastic gradient descent to adjust values of input image to maximize the
#  - activation value

# let's try on block 3 of the vgg16 net

model <- application_vgg16(include_top = FALSE, weights = "imagenet")

layer_name <- "block3_conv1"
filter_index <- 1

layer_output <- get_layer(model, layer_name)$output

# NOTE: use of k_functions. these are tensorflow backend functions for low level
# model manipulation.

loss <- k_mean(layer_output[, , , filter_index])

# run gradient descent on loss with respect to input
# (this returns list of tensors. size one in this case, so we just take first element)
grads <- k_gradients(loss, model$input)[[1]]
 # alternatiely, we might want to normalize the gradient tensor with L2 norm.
# this ensures that magnitudes of updates on images stays relatively even
grads <- grads / k_sqrt(k_mean(k_square(grads)) + 1e-5) # +1e-05 prevents division by 0


# we now need to compute value of thhe loss tensor and
# the gradient tensor, given an input image. 

# iterate() will take a tensor (input) and return list of two tensors (loss and gradient)
iterate <- k_function(list(model$input), list(loss, grads))

c(loss_value, grads_value) %<-%
  iterate(list(array(0, dim = c(1, 150, 150, 3))))

# now run loop for gradient descent

# create gray image with noise:
input_img_data <-
  array(runif(150*150*3), dim = c(1, 150, 150, 3)) * 20 + 128 # <- starts from a gray image with noise

step <- 1
for(i in 1:40) {
  
  # calcualte loss and gradient tensors for that input image
  c(loss_value, grads_value) %<-%
    iterate(list(input_img_data))
  
  # update image in direction of gradient by chosen step size
  input_img_data <- input_img_data + (grads_value * step)
  
}

dim(input_img_data)
max(input_img_data)

# output doesn't fall within 0-255 so we need to post-process it
deprocess_image <- function(x) {
  
  dms <- dim(x)
  
  # normalize image to mean ~0 and sd 0.1
  x <- x - mean(x)
  x <- x / (sd(x) + 1e-5)
  # get sd of .1
  x <- x * 0.1
  
  # clip to [0,1]
  x <- x + 0.5
  x <- pmax(0, pmin(x, 1))
  
  array(x, dim = dms)
}


# now lets create one function that takes a layer and filter index and returns representative tensor
generate_pattern <- function(layer_name, filter_index, size = 150) {
  
  layer_output <- model$get_layer(layer_name)$output
  loss <- k_mean(layer_output[, , , filter_index])
  
  grads <- k_gradients(loss, model$input)[[1]]
  
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  iterate <- k_function(list(model$input), list(loss, grads))
  
  input_img_data <- 
    array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  
  step <- 1
  for(i in 1:40) {
    
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    
    input_img_data <- input_img_data + (grads_value * step)
    
  }
  
  img <- input_img_data[1, , , ]
  deprocess_image(img)
  
}

# lets give it a go:
library(grid)

grid.raster(generate_pattern("block3_conv1", 1))



# now lets scale this and see many filters from many layers (first 64 filters
# for the first layer in each convolutional block)

library(gridExtra)

dir.create("~/Documents/deep-learning-with-r/vgg_filters")

for(layer_names in c("block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1")) {
  
  size <- 140
  
  png(paste0("~/Documents/deep-learning-with-r/vgg_filters/", layer_name, ".png"),
      width = 8 * size, height = 8 * size)
  
  grobs <- list()
  for(i in 0:7) {
    
    for(j in 0:7) {
      
      pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
      grob <- rasterGrob(pattern,
                         width = unit(0.9, "npc"),
                         height = unit(0.9, "npc"))
      
      grobs[[length(grobs) + 1]] <- grob
      
    }
    
  }
  
  grid.arrange(grobs = grobs, ncol = 8)
  dev.off()
  
}


# 3. Visualizing heatmaps of classifications
#   great for understanding how a net comes to its classification decisions and debugging
# AKA Class-Activation Map (CAM)

# really producing a 2d array of classification scores overlayed onto the input image

# in technical terms, we are taking the output feature map of a layer, given some
# input, and weighing each channel by the gradient of the class with respect to that channel.
# in other words, we are weihting a spatial map how how intensley input images activate different channels
# by how important each channel is for that class. This creates a map of how intensely the input
# image is activating each channel.

model <- application_vgg16(weights = "imagenet")

# rather than use the elephant image in the book, we'll keep using a cat image at img_path
img_path

# preprocess for size 224 * 224 for the net
img <- image_load(img_path, target_size = c(224, 224)) %>% 
  image_to_array() %>% 
  array_reshape(c(1, 224, 224, 3)) %>% 
  imagenet_preprocess_input()

# run image thourh the network and decode its prediction vector to something readable
preds <- model %>% 
  predict(img)

imagenet_decode_predictions(preds, top = 3)

# it classified as egyption cat with 53% probability

# now lets setup the grad-CAM process
# check index in the prediction vector (length num.classifications)
which.max(preds[1,])
#286

# get that egpytian cat entry in the prediction vector
egyptian_cat_output <- model$output[, 286]

# get output feature map of the conv. last layer in the net
last_conv_layer <- model %>% get_layer("block5_conv3")

# gradient of egyptian cat class with regard to the output feature map of final layer
grads <- k_gradients(egyptian_cat_output, last_conv_layer$output)[[1]]

pooled_grads <- k_mean(grads, axis = c(1, 2, 3))

iterate <- k_function(list(model$input), list(pooled_grads, last_conv_layer$output[1, , , ]))

c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))

for(i in 1:512) {
  
  conv_layer_output_value[, , i] <-
    conv_layer_output_value[, , i] * pooled_grads_value[[i]]
  
}

# channel-wise average (of filters) of our feature map is the heatmap of the class activation
heatmap <- apply(conv_layer_output_value, c(1, 2), mean)

# normalize to visualze
heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)

# function to write heatmap to a png
write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
  
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0, 0, 0, 0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
  
}

write_heatmap(heatmap, "~/Documents/deep-learning-with-r/dogs-vs-cats/cat_heatmap.png")


# now use magick package to superimpose the heatmap onto the original image
library(magick)
library(viridis)

# read original image and its geometry
img <- image_read(img_path)
info <- image_info(img)
geometry <- sprintf("%dx%d!", info$width, info$height)

# create blended/transparent version of the heatmap
pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)

write_heatmap(heatmap, "~/Documents/deep-learning-with-r/dogs-vs-cats/cat_overlay.png",
              width = 14, height = 14, bg = NA, col = pal_col)

# read and display (plot)
image_read("~/Documents/deep-learning-with-r/dogs-vs-cats/cat_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(img, operator = "blend", compose_args = "20") %>% 
  plot()
