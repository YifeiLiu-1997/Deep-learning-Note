# GAN: Generative Adversarial Network
1. collect real image (only discriminator will see)
2. input some random noise into generator and generator will generate fake data (audio, image, viedo, etc.)
3. discriminator tell's fake or real and be more strong
4. so games begin, adversarial begin until generator is strong enough

## Notation
theta: generator parameters\
Noise: Input

## Normal Classifier
![image](https://user-images.githubusercontent.com/71109255/124405473-527e8b80-dd71-11eb-97b9-66a031504456.png)
- and discriminator is two class classifier (fake or real)

## generator learning
![image](https://user-images.githubusercontent.com/71109255/124405418-2f53dc00-dd71-11eb-8ec6-f85838873b27.png)
- generator want y_hat from 0 - 1, make it real, discriminator want y_hat from 1 - 0, classifier generator's result to fake

## BCE Cost Function: Binary Cross Entropy function
![image](https://user-images.githubusercontent.com/71109255/124405833-392a0f00-dd72-11eb-85ca-36480622b929.png)

## Training Both
1. firtst, use BCE training disciminator's theta
2. second, use BCE training generator's theta
3. to sum, d and g prove together, so do not use really good generator or really good discriminator

