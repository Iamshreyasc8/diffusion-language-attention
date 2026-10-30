# Diffusion Language Attention

Research notebooks exploring how words and multi-word expressions relate to spatial attention in Stable Diffusion using DAAM (Diffusion Attentive Attribution Maps).

The project combines text preparation, image generation, attention visualization, and pairwise heatmap overlap analysis. These are exploratory research notebooks, not a validated end-to-end application.

## Notebooks

| Notebook | Contents |
| --- | --- |
| `notebooks/Daam_run_Model.ipynb` | Stable Diffusion generation, word attention overlays, attention-head visualization, and prompt variation experiments. |
| `notebooks/token.ipynb` | Dataset plots, token counting, and prompt rewriting experiments using T5/FLAN-T5 and BART. |
| `notebooks/error.ipynb` | Experimental multi-word expression analysis, binary attention masks, and continuous pairwise heatmap IoU. Includes debugging and alternative setup cells. |

## Getting started

1. Open a notebook in Jupyter or Google Colab. Several cells assume Colab and a CUDA GPU.
2. Inspect the installation cells before running. The notebooks contain different historical dependency pins and alternative setups; they should not be installed together blindly. No shared dependency environment has been validated.
3. Supply the datasets referenced by each experiment and adapt the `/content/drive/MyDrive/Thesis/...` paths. Dataset files, model weights, and generated results are not included.
4. For Hugging Face authentication, run the provided `login()` call and enter your own token interactively. Do not paste credentials into notebook source or commit them.
5. Run the relevant experiment cells in order after resolving their inputs and setup. Some experiments load multiple large models; use separate sessions where needed.

Depending on the experiment, packages include PyTorch, Diffusers, Transformers, DAAM, Hugging Face Hub, Accelerate, NumPy, pandas, Matplotlib, Seaborn, SciPy, OpenCV, OpenPyXL, SentencePiece, and tqdm. Follow each model's access and licensing requirements.

## Analysis approach

The attention experiments generate an image under DAAM tracing and extract word-level heatmaps. Multi-word expression analysis compares token heatmaps using continuous overlap:

`IoU(A, B) = sum(min(A, B)) / sum(max(A, B))`

The code returns zero when the denominator is zero. Percentile-thresholded masks provide a separate visualization. Attention overlap is an exploratory signal; it does not by itself establish semantic understanding or linguistic compositionality.

## Status and limitations

- Source notebooks have been preserved with embedded Hugging Face tokens removed and all saved outputs and execution counts cleared.
- Original experiment logic and dependency versions have not been modernized or verified by executing model inference.
- `error.ipynb` contains debugging work and is not presented as a working production pipeline.
- Some code paths and missing-token handling may need fixes before execution.
- No numerical results or performance claims are asserted by this repository.

## Acknowledgements

These experiments use third-party models and libraries, including [DAAM](https://github.com/castorini/daam), Hugging Face Diffusers and Transformers, Stable Diffusion, FLAN-T5, and BART. Their code, models, and datasets retain their respective licenses.
