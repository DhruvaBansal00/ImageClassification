Name: Dhruva Bansal
Email ID: dbansal36@gatech.edu
Best Accuracy: 83% on testing data. 

Note that all training for the transferLearning models was done on google colab using the code in transferLearning.py. 

I initially tried using multiple convolution layers along with a fully connected layer to classify. Unfortunately, I was unable to get above 65% accuracy on the test data using this method. Hence, I switched to transfer learning using various pretrained networks. Since these networks work well only on images of size 224x224, I upscaled all the inputs to that size. I first tried vgg16 and vgg19 and was able to get 67% accuracy. I then switched to resnet50 and got around 70% accuracy. After researching a little more, I realized that resnet152 may give better features and got 83% accuracy using that. I then also tried data augmentation by randomly flipping certain images and cropping them after padding with 0s. This increased the accuracy to 86%. However, I forgot to save the .log files and hence wasn't able to create the graphs for this net. Hence, all the graphs and models are for the one with 83% accuracy. I did however save the google colab output for the 86% accuracy model and have included that in my submission. I also tried transfer learning with resnext50 and resnext101 but they didn't give much improvement on accuracy. 


