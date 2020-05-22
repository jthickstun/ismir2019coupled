# ismir2019coupled
Experiments for [Coupled Recurrent Models for Polyphonic Music Composition](https://arxiv.org/abs/1811.08045).

Corresponding author: [__John Thickstun__](https://homes.cs.washington.edu/~thickstn/). University of Washington

-------------------------------------------------------------------------------------

This repository contains code for the experiments discussed in Coupled Recurrent Models for Polyphonic Music Composition.

Included are notebooks for reproducing the single-part and multi-part experiments presented in Table 3 and Table 4 respectively.

For examples of how to generate output from the model in audio, kern, and midi representations, see sampling.ipynb. A pre-trained model is included for the multipart6 architecture.

## Dependencies

Dependencies are inherited from the NCSN and Glow projects. For NCSN:

* PyTorch (tested with v1.5.0)

* Jupyter (tested with v4.4.0)

* NumPy (tested with v1.18.4)

* SciPy (tested with v1.4.1)

* Mido (tested with v1.2.9)

* Scikit-learn (tested with v0.22.2)

* Matplotlib (tested with v2.1.2)

## References

To reference this work, please cite

```bib
@article{thickstun2019coupled,
  author    = {Thickstun, John and Harchaoui, Zaid and Foster, Dean P and Kakade, Sham M},
  title     = {Coupled Recurrent Models for Polyphonic Music Composition},
  journal   = {International Society for Music Information Retrieval},
  year      = {2019},
}
```

