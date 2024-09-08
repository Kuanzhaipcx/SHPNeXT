# SHPNeXT
SHPNeXT: Enhanced Tongue Image Segmentation Across Multi-scale and Variable Resolutions for Traditional Chinese Medicine

The relevant experimental code of this paper relies on the open source library 

```
MMsegmentation
```

. Please download and follow the corresponding tutorial to run and compare the experimental code. The method code mentioned in this article has been put into this compressed package. Put 

```
segfinal_backbone.py
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

in this compressed package. Due to privacy restrictions, we are unable to provide the 

```
HUCM 
```

data set.
