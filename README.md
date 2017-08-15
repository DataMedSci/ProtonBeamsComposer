# ProtonBeamsComposer

### Under development...

[![wercker status](https://app.wercker.com/status/e724af14246bc2e21ee83eded6d0729e/m/ "wercker status")](https://app.wercker.com/project/byKey/e724af14246bc2e21ee83eded6d0729e)

### Example usage

From the main directory execute line below to generate a SOBP with range 15.0 mm and 15.0 mm spread.

```
$ python run.py --range 15 --spread 15 --verbose 
```

Full range and full spread without plots, but with verbose logging:

```
$ python run.py -r 1 -s 1 -vnf both
```

Use external file with BP data `data\cydos1.dat` and generate a range 25.0 mm and spread 11.0 mm SOBP with quiet plotting.

```
$ python run.py -i data\cydos1.dat -r 25 -s 11 -n
```

### Available parameters for `run.py`

```
Required:
-s, --spread [float]
-r, --range [float]

Advanced:
-n, --name [str] (name for output directory)
-f, --full ['range', 'spread', 'both'] (generation options, result based on input file)
-p, --halfmod (use half range as modulation value)
    --smooth (use Savgol filter to smooth input data)
    --window [int, odd] (specify window used by Savgol filter)
-g, --add_to_gott [int] (add this number of peaks to calculated with Gottschalk rule, can be negative)
-k, --peaks [int] (number of peaks in optimization - omits calculation using Gottschalk rule)
-i, --input_bp_file [str] (file with BP data, first two columns should contain domain and values respectively)
-d, --delimiter [str] (delimiter used in BP file)

Logging etc.:
-v, --verbose (DEBUG logging level)
-q, --quiet (do not print anything below WARNING level)
-n, --no-plot (disable showing most of the plots)
```
