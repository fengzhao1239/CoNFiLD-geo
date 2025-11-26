# Conditional Neural Field Latent Diffusion for Geoscience

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.06525-b31b1b.svg)](https://arxiv.org/abs/2508.16640)

<p align="center">
  <img src="assets/pic1.png" alt="framework" width="20%">
</p>
Conditional Neural Field Latent Diffusion for Geoscience (CoNFiLD-geo) is a generative framework for zero-shot conditional reconstruction of geomodels and corresponding reservoir responses from diverse types of observational data, enabling real-time inversion with unceetainty quantification in realistic geological carbon sequestration projects.

## 🛠️ Installation

### create a conda environment named "CoNFiLD"

1. install `conda` managed packages
    ```bash
    conda env create -f env.yml
    ```
2. change `conda` environment
    ```bash
    conda activate CoNFiLD
    ```
3. install `pip` managed packages
    ```bash
    pip install -r requirements_pip.txt
    ```
### Using python Environment
* Create a `.env` file in the CoNFiLD directory and copy the following settings within the file
    ```bash
    PYTHONPATH=./:UnconditionalDiffusionTraining_and_Generation:ConditionalNeuralField:$PYTHONPATH
    CUDA_VISIBLE_DEVICES= #set your GPU number(s) here
    ```

* Run the following bash command
    ```bash
    set -o allexport && source .env && set +o allexport
    ```
  
## 🚀 Using pretrained CoNFiLD-geo

### Download pretrained model
* The trained model parameters associated with this code can be downloaded [here](https://drive.google.com/drive/folders/1eEnaE50Z7xN3HQevK-nlcO2-CB-0384r?usp=sharing)

### Generating Conditional Samples
* Here we provide the conditional generation script for an example case.
    * The conditional generation template can be found in `ConditionalDiffusionGeneration/inference_scripts/template.ipynb`
    * For creating your arbitrary conditioning, please define your forward function in `ConditionalDiffusionGeneration/src/guided_diffusion/measurements.py`
    
* To understand the conditional generation process, please follow the instructions in the Jupyter Notebook `ConditionalDiffusionGeneration/inference_scripts/Cartesian_inverse.ipynb`. You can play with the classes in measurements.py by specifying your own conditional inputs. All the functions to reproduce the results in the manuscript are included in this measurements.py file.
    
## 🔥 Training CoNFiLD-geo from scratch

### Download data
* The dataset associated with this code can be downloaded [here](https://drive.google.com/drive/folders/1Q_EXqvaeNVU8Rc-Br7nHGDuxfnndQg6x?usp=drive_link)
  
### Training Conditional Neural Field
* Use `train.py` under `ConditionalNeuralField/scripts` directory
    ```bash
    python ConditionalNeuralField/scripts/train.py PATH/TO/YOUR/xxx.yaml
    ```
        
### Training Diffusion Model
* After the CNF is trained: 
    * Process the latents into square images with dimensions of the square equal to the latent vector length
    * Add a channel dimension after the batch dimension. The final shape should be $(B\: 1\: H\: W)$
* Use `train.py` under `UnconditionalDiffusionTraining_and_Generation/scripts` directory
    ```bash
    python UnconditionalDiffusionTraining_and_Generation/scripts/train.py PATH/TO/YOUR/xxx.yaml
    ```

## 💡 Acknowledgement
The diffusion model used in this work is based on [OpenAI's implementation](https://github.com/openai/guided-diffusion). The DPS part is based on [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://github.com/DPS2022/diffusion-posterior-sampling). This code repo was modified from the [original CoNFiLD repo](https://github.com/jx-wang-s-group/CoNFiLD). 







