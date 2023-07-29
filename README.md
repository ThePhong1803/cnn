# Handwritten digit recognition 
Handwritten digit recognition, using a very basic neural network

# Note:
- Using library Eigen 3.4.0.
- For reading image input, I wrote my Image class that support 8-bit and 24-bt bitmap file, and convert the image to gray scale.
- For testing, using paint and open the image in /test/test.bmp, redraw the image and then hit enter key.
- To exit test loop, type "exit" and hit enter.
- For easy while using paint, the setInvert() take True value. For the image is the same as type as training data, clear invert flag.
# Training and testing result
Training RMSE, Validate RMSE and accuracy over epoch (learning rate = 0.1, epoch = 9000, batch = 100)
![alt text](https://github.com/ThePhong1803/handwritten/blob/master/Rplotter/train_result.png)

# Update with convolutional layer
- Divide the neural network into several layer and implement it separately
- Implemented layers: convolutional layer, max pooling layer, flatten layer
- Comming soon: Dense layer, Average pooling layer, Softmax layer,...

# Reference materials
https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/
https://www.youtube.com/watch?v=k_VdZVJeEyg
