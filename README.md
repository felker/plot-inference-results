# plot-inference-results

Run `ln -s -f ../../.githooks/pre-commit .git/hooks/pre-commit` in the main directory of
the repository to install a hook that clears Jupyter notebook outputs before committing
changes.


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
