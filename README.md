# Conditional Neural Field Latent Diffusion for Geoscience

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.06525-b31b1b.svg)](https://arxiv.org/abs/2508.16640)

<p align="center">
  <img src="assets/pic1.png" alt="framework" width="20%">
</p>
Conditional Neural Field Latent Diffusion for Geoscience (CoNFiLD-geo) is a generative framework for zero-shot conditional reconstruction of geomodels and corresponding reservoir responses from diverse types of observational data, enabling real-time inversion with unceetainty quantification in realistic geological carbon sequestration projects.

## Create python Environment

create a conda environment named "CoNFiLD-geo"

1. install `conda` managed packages
    ```bash
    conda env create -f env.yml
    ```
2. change `conda` environment
    ```bash
    conda activate CoNFiLD-geo
    ```
3. install `pip` managed packages
    ```bash
    pip install -r requirements_pip.txt
    ```
## Using python Environment
* Create a `.env` file in the CoNFiLD-geo directory and copy the following settings within the file
    ```bash
    PYTHONPATH=./:UnconditionalDiffusionTraining_and_Generation:ConditionalNeuralField:$PYTHONPATH
    CUDA_VISIBLE_DEVICES= #set your GPU number(s) here
    ```

* Run the following bash command
    ```bash
    set -o allexport && source .env && set +o allexport
    ```
  
## Using pretrained CoNFiLD-geo

### Download pretrained model
* The trained checkpoints (CNF, diffusion model and UFNO) can be downloaded from [Hugging Face](https://huggingface.co/zhaoffeng/CoNFiLD-geo). Place them in the repository root, keeping the directory structure.

### Generating Conditional Samples
* The conditional generation notebooks are provided for each case under `ConditionalDiffusionGeneration/inference_scripts`
    * For creating your arbitrary conditioning, please define your forward function in `ConditionalDiffusionGeneration/src/guided_diffusion/measurements.py`

* Run the notebooks under `ConditionalDiffusionGeneration/inference_scripts/{Cartesian,Egg,Norway,Topography}` from their own directory. Each notebook loads
    * the trained CNF checkpoint and `normalizer_params.pt` from `ConditionalNeuralField/output_*/`
    * the trained diffusion model (`ema_*.pt`) from `UnconditionalDiffusionTraining_and_Generation/output/`
    * the data from `Dataset/`
* Results are written to the `*_outputs` directory next to each notebook
* Unconditional sampling is also available via `UnconditionalDiffusionTraining_and_Generation/scripts/inference.py`
    
## Training CoNFiLD-geo from scratch

### Training Conditional Neural Field
* Use `train.py` under `ConditionalNeuralField/scripts` directory
    ```bash
    python ConditionalNeuralField/scripts/train.py PATH/TO/YOUR/xxx.yaml
    ```

* Download the corresponding case data from [Hugging Face](https://huggingface.co/datasets/zhaoffeng/CoNFiLD-geo) and add it in the `Dataset` directory, and use the `ConditionalNeuralField/training_recipes/{Cartesian_simple,EggChannelized,Norway,Topography}_joint.yml`
    * Data and query coordinates are loaded by the `load_data_fn` and `load_coord_fn` specified in the yaml file (see `ConditionalNeuralField/cnf/utils/readdata.py`)
    * Checkpoints (`checkpoint_*.pt`, which also store the trained latents) and `normalizer_params.pt` are saved to `save_path`
        
### Training Diffusion Model
* After the CNF is trained, the latents are read directly from the CNF checkpoint given by `latent_checkpoint_path` in the yaml file
    * `UnconditionalDiffusionTraining_and_Generation/latents/create_dataset.py` reshapes the latents into images of shape $(B\: 1\: T\: D)$, splits them into training/validation sets and normalizes them to $[-1, 1]$
    * Select the dataset class for your case in `UnconditionalDiffusionTraining_and_Generation/scripts/train.py`
* Use `train.py` under `UnconditionalDiffusionTraining_and_Generation/scripts` directory
    ```bash
    python UnconditionalDiffusionTraining_and_Generation/scripts/train.py PATH/TO/YOUR/xxx.yaml
    ```
* Use the yaml files under `UnconditionalDiffusionTraining_and_Generation/training_recipes` (`cartesian_simple_diff.yml`, `Egg_model.yml`, `Norway_3D.yml`, `Topography.yml`)
    * Checkpoints (`model*.pt`, `ema_*.pt`) are saved to `log_path`

## Acknowledgement
The diffusion model used in this work is based on [OpenAI's implementation](https://github.com/openai/guided-diffusion). The DPS part is based on [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://github.com/DPS2022/diffusion-posterior-sampling).







