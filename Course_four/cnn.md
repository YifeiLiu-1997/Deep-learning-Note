# started by edge detect
- how to detect the edge (vertical) of image 6×6×1 (gray scale image)
- to build a filter 3×3×1 and do a convolution operation

# convolution operation
- [image here]
# pooling
- max pooling means

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
- AlexNet: 
- VGG-16: 
