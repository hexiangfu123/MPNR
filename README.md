# MPNR

## Setup

### Requirements
```Python==3.11```

```Pytorch==2.0.1```

```transformers==4.34.1```

```scikit-learn==1.3.2```

```tqdm==4.65.0```

```fire==0.5.0```

### Dataset
We validate the effectiveness of our model in the real world dataset [MIND](https://msnews.github.io/)

___
## Training

single GPU (cuda:0)
```
sh run1.sh debug 0 MPNR mind_small
```

multi GPU (cuda:0,1,2,3)
```
sh run1.sh run 0,1,2,3 MPNR mind_small
```