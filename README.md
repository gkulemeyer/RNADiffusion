# RNADiffusion

RNADiffusion is a diffusion-based **seq2struct model** for **RNA secondary structure prediction**.

The model takes an RNA sequence as input, builds a 2D representation via an outer product of sequence concatenates it with 2D noise, and uses a **diffusion model** to generate a contact map representing RNA secondary structure.

This repository is intended for model development, training, and experimentation.

---

## TODO

- Add gumbel_softmax
- Check padding consistency
- Check loss implementation and evaluation (one timestep / every timestep / stochastic timesteps )
- Add evaluation scripts to see results on every timestep, to handle the statistics of generative results. 