# started by edge detect
- how to detect the edge (vertical) of image 6×6×1 (gray scale image)
- to build a filter 3×3×1 and do a convolution operation

# convolution operation
![image](https://user-images.githubusercontent.com/71109255/124054912-872ed200-da55-11eb-8f28-228f458e6db6.png)
# pooling
![image](https://user-images.githubusercontent.com/71109255/124056752-e3472580-da58-11eb-9a4c-047f26479282.png)

# some notation (l layer)
- filter size = f[l]
- padding: p[l]
- stride: s[l]
- input shape: nh[0] × nw[0]× nc[0]
- activations: a[l], shape: output shape
- each filter shape: f[l] × f[l] × nc[l-1]
- weights: each filter shape × nc[l]
- bias: nc[l] - (1, 1, 1, nc[l])
- output shape: nh[l] × nw[l] × nc[l]
- iter: nh[l] = floor(nh[l-1]+2×p[l]-f[l]/s[l] + 1)
- iter: nw[l] = floor(nw[l-1]+2×p[l]-f[l]/s[l] + 1)

# AlexNet, LeNet - 5, VGG - 16
- LeNet-5: 1998
![image](https://user-images.githubusercontent.com/71109255/124053713-4e8df900-da53-11eb-856c-7cddb507eaa9.png)
- AlexNet: 2012
![image](https://user-images.githubusercontent.com/71109255/124054445-98c3aa00-da54-11eb-8bfa-640051719b8f.png)
- VGG-16: 2015
![image](https://user-images.githubusercontent.com/71109255/124054852-6cf4f400-da55-11eb-9460-05b577f06ce1.png)

