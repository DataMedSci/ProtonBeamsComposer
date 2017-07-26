# ProtonBeamsComposer

### Under development...

[![wercker status](https://app.wercker.com/status/e724af14246bc2e21ee83eded6d0729e/m/ "wercker status")](https://app.wercker.com/project/byKey/e724af14246bc2e21ee83eded6d0729e)

### Example usage

From main directory execute line below to generate a SOBP with range 15.0mm and 15.0mm spread.

```
$ python run.py --range 15 --spread 15 --verbose 
```

Another example generating full range and full spread without plots, but with verbose logging:

```
$ python run.py -r 1 -s 1 -vnf both
```

### Available parameters

```
-s, --spread [float]
-r, --range [float]
-f, --full ['range', 'spread', 'both']
-v, --verbose (DEBUG logging mode)
-q, --quiet (do not print anythink below WARNING level)
-p, --halfmod (use half range as modulation value)
-n, --no-plot (disable showing most of the plots)
```
