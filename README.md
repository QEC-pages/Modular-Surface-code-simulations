# Modular Surface Code Simulations

This repository contains the simulation code accompanying our paper:

**"Optimized Noise-Resilient Surface Code Teleportation Interfaces"**  
[arXiv:2503.04968](https://arxiv.org/abs/2503.04968)

The purpose of this repository is to facilitate reproducibility and further exploration of our results. Please feel free to contact us with any questions or feedback.

---

## Repository Structure

- **`lib/`**  
  Contains functions for generating Stim circuits. Each file corresponds to a specific configuration studied in our paper:
  - Rotated vs. Unrotated Surface Codes
  - Interface Gadgets: Direct, CAT, and Gate Teleportation (GT)

  (Note: Some redundancy is present for clarity and ease of use.)

- **`Simulation.ipynb`**  
  Jupyter notebook responsible for sampling from the generated circuits and performing decoding using PyMatching.

- **`Analysis and Plotting.ipynb`**  
  Jupyter notebook dedicated to analyzing simulation data. This includes:
  - Threshold calculations
  - Gamma factor computations
  - Visualization and plotting of results

- **Raw Data**  
  - **`[[Feb13th]] Full Results.json`**: Complete raw data from simulations presented in our paper.
  - **`thresholds.csv`**: Threshold values derived from the simulation data.
  - **`lambda p=0.00345.csv`**: Specific lambda calculations at physical error rate p=0.00345.

- **Plots/**  
  Directory containing high-quality PDF plots generated for the publication.

---

## Usage

To reproduce results, we recommend starting with `Simulation.ipynb` to generate or validate existing data, followed by `Analysis and Plotting.ipynb` for detailed analysis and visualization.

---

## Citation

If you use this code or find our work useful, please cite:

```
@article{your_citation_here,
  title={Optimized Noise-Resilient Surface Code Teleportation Interfaces},
  author={Authors},
  journal={Journal or Conference},
  year={2025},
  eprint={2503.04968},
  archivePrefix={arXiv},
  primaryClass={quant-ph}
}
```

---

For any additional information, please don't hesitate to reach out.
