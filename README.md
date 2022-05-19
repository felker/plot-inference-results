# plot-inference-results

Run `ln -s -f ../../.githooks/pre-commit .git/hooks/pre-commit` (yes, `../../` is the correct relative path even when `ln` is run from top-level repository directory, since it is relative to `.git/hooks/`) in the main directory of the repository to install a hook that clears Jupyter notebook outputs before committing changes.


## Batch size = 1 observations:
- Runtime grows linearly with the length of the input sequence
- Runtime grows (probably) linearly with the input batch size
- Only interesting pattern are the kinks in the `num_hidden=200, 256` lines:
  - The runtime starting from the `num_hidden=200 seq_length=200` case grows superlinearly, and the `num_hidden=256 seq_length=200` case grows sublinearly, so that the runtime of the `num_hidden=200 seq_length=256` and `num_hidden=256 seq_length=256` cases are inverted w.r..t expectations (the larger problem is faster)
  - Likely due to "nice" power-of-2 and/or square matrix sizes being amenable to A100 TensorCore usage. 
  - Only appears for `num_layers >= 3`, at large input sequence length
  - Doesn't appear for `num_layer=2` likely because only starting with the second layer do the input weight matrices become `(batch_size, num_hidden, num_hidden)`. E.g. examining the ONNX file, the 2x weight matrices (each unified acrossall 4x gated matrices) for the input and recurrent state, respectively, are for the first layer:
    ```
    W (1x800x12)
    R (1x800x200)
    ```
    They are the same size for subsequent LSTM layers.


## Batch size > 1 observations:
- we can pack a decent batch size on the A100 while the runtime only grows slowly and linearly. It would be cool to measure SM occupancy to show when that trend breaks down (e.g. hidden size=400, 300, 256, 200 at batch size=256)
- the runtime grows much more quickly the number of layers and input sequence length, which makes sense since those dimensions cannot fully be parallelized. But we can get very accurate models using only 2 LSTM layers and a reasonable input sequence length. 
-  there is a definite sweet spot for batch size=2 to 4
-  for the batch_size=1 case in the larger LSTMs— if they are truly slower than the batch_size=2 case, then it would make sense to simply duplicate the input along the batch size just to get a faster runtime. But this could theoretically / should be done under the hood in the TensorRT engine creation, so I am suspicious. 


As expected, the distribution of inference times is multimodal— narrowly peaked around the median runtime, with some significant clusters of much longer inference times. 

The boxplots are so highly compressed around the interquartile range that you can’t make out the boxes and whiskers, but up to ~10% of the samples are outliers (I have manually annotated the number of outliers out of the 1000 trials). The outliers are highly clustered around a couple of runtimes, e.g. 1.25x and 2x median runtime— if we dug into them a little bit, we could probably explain these factors with CUDA kernel scheduling behaviors. 

This is probably similar to how real-time operating systems characterize the latencies of non-RT OS system latencies. 
> Usually, we take a draconian but simplistic approach: Max time rules. Ignore min, ignore average.
> Sometimes, we throw out the first couple times if we can guarantee they can be done outside of the time of interest.
