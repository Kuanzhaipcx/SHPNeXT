# SHPNeXT
SHPNeXT: Enhanced Tongue Image Segmentation Across Multi-scale and Variable Resolutions for Traditional Chinese Medicine


This repository only stores relevant code for the methods proposed in this paper.
Another experimental code of this paper relies on the open source library 

```
MMsegmentation: https://github.com/open-mmlab/mmsegmentation.git
```

. Please download and follow the corresponding tutorial to run and compare the experimental code. The method code mentioned in this article has been put into this compressed package. Put 

```
SHPNeXt_backbone.py
```

 and the corresponding 

```
poolformer
```

 and 

```
hire-mlp
```

 scripts into the 

```
backbone
```

 folder in mmsegmentation. Put 

```
ham_head_nuclearnmf_decoder.py 
```

into the decoder folder. Follow the mmsegmentation tutorial. Register and configure settings and you're ready to run.

The two open source data sets 

```
BioHit
```

 and 

```
LRCM 
```

used in this article have been put into the 

```
data 
```

in this compressed package. You can download it from the link ``` https://drive.google.com/file/d/1CTfb5x9I79FUreqRgssu01hqnsjh8h6i/view?usp=drive_link ```. Due to privacy restrictions, we are unable to provide the 

```
HUCM 
```

data set.
