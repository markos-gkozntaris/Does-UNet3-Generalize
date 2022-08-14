# Does UNet3 Generalize?

Project for TU Delft's CS4245 course "Seminar Computer Vision by Deep Learning"

Blogpost about this project: https://nickgavalas.com/ablation-study-of-unet3-on-ct-scans/

#### Implementation notes about the vanilla Unet

In this version Batch Normalization is used which is absent in the original paper.

Also, the original paper crops the image before concatenating, while here if pad = "pad" it performs same padding, else it crops.
