xMIL: Insightful Explanations for Multiple Instance Learning in Histopathology
==========

<details>
<summary>
  <b>A 3-class classifier to identify metastatic and lymphomatous lymph nodes</b>. EMBC 2026.
  <br><em>Stéphane Treillard, Robin Schwob, Raphaëlle Duprez-Paumier, Philippe Rochaix, Charlotte Syrykh, Camille Laurent, Pierre Brousset, Sandrine Mouysset, Nadia Amara, Sylvain Cussat-Blanc, Camille Franchet</em></br>
  
  This repo is forked from https://github.com/bifold-pathomics/xMIL, from the wonderful:
  <b>xMIL: Insightful Explanations for Multiple Instance Learning in Histopathology</b>. NeurIPS 2024.
  <br><em>Julius Hense*, Mina Jamshidi Idaji*, Oliver Eberle, Thomas Schnake, Jonas Dippel, Laure Ciernik, 
Oliver Buchstab, Andreas Mock, Frederick Klauschen, Klaus-Robert Müller </em></br>
* Equal contribution


</summary>

```
@inproceedings{hense2024xmil,
  author = {Hense, Julius and Jamshidi Idaji, Mina and Eberle, Oliver and Schnake, Thomas and Dippel, Jonas and Ciernik, Laure and Buchstab, Oliver and Mock, Andreas and Klauschen, Frederick and M\"{u}ller, Klaus-Robert},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {8300--8328},
  publisher = {Curran Associates, Inc.},
  title = {xMIL: Insightful Explanations for Multiple Instance Learning in Histopathology},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/0f9e0309d8a947ca44463a9b7e8b6a3f-Paper-Conference.pdf},
  volume = {37},
  year = {2024}
}
```

</details>

<p align="center">
  <img src="xMIL.png"/>
</p>


**Summary**: In this study, we train several models to classify benign, metastatic and lymphomatous WSIs, with a focus on explainability. We obtain high accuracies on an in-house dataset and on several external datasets. 

## Installation
For installation:
- Create an environment: ```conda create -n "xMIL-BeMeLy" python=3.9```, and activate it ```conda activate xMIL-BeMeLy```
- Cloning: ```git clone https://github.com/mahmoodlab/trident.git && cd xMIL-BeMeLy```
- Local installation: ```bash install_requirements.sh```

## Downloading the validation datasets
Our models were validated on CAMELYON16, CAMELYON17, TCIA-DLBCL, TCIA-SLN-Breast and on the IICBU lymphoma dataset

### CAMELYON
CAMELYON16 and CAMELYON17 are a collection of benign and metastatic axillary lymph nodes. The datasets were  introduced in **1399 H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset, Litjens et al., GigaScience, 10.1001/jama.2017.14585** and **Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer, Bejnordi et al., JAMA. 10.1093/gigascience/giy065**, respectively. To download slides from both CAMELYON datasets, see https://camelyon17.grand-challenge.org/Data/

### TCIA-DLBCL
DLBCL is a high-grade lymphoma that is relatively common. To download DLBCL slides from TCIA, see https://www.cancerimagingarchive.net/collection/dlbcl-morphology/

### TCIA-SLN-Breast
To download this collection of benign and metastatic axillary lymph nodes, see https://www.cancerimagingarchive.net/collection/sln-breast/

### IICBU lymphoma
This dataset was introduced in **IICBU 2008: a proposed benchmark suite for biological image analysis, Shamir et al., Med Biol Eng Comput, 10.1007/s11517-008-0380-5**. It contains tiles extracted from three different lymphoma subtypes: FL, MCL and SLL. This dataset can no longer be downloaded from the original article, but the tiles are still available here: https://andrewjanowczyk.com/deep-learning/

## Feature extraction
Feature extraction was performed using Virchow2 and Uni-2h as patch feature extractors, using trident: https://github.com/mahmoodlab/TRIDENT

## Model training
To train models as was done in the article, create a csv in the form of ```dummy_template.csv``` containing slide names and patient ids. Create another csv containing split information, in the form of ```splits/splits_0.csv``` (create several if you want to do cross-validation or train models across multiple train-test splits). Then run ```bash train.sh```, replacing all paths to paths to your data. To avoid the patterns in explainability heatmaps that were encountered in the article, make sure that ```no_attn_residual``` is set to True in ```train_model.py``` (in the article it was set to False).

## Testing
To test the models, simply run ```bash test.sh```, replacing paths to paths to your data.

## Assessing faithfulness
Once you have trained a model, run ```python flip_patches.py``` to benchmark multiple patch importance attribution methods. You can then check results in ```notebooks/patch_flipping_plots.ipynb```

## Generating heatmaps
To generate explanation heatmaps, see ```notebooks/slide_visualizations_compute_heatmaps.ipynb```

## Statistical testing
To perform statistical testing to identify the best performing model, see ```notebooks/stats_test.ipynb```

## Reproducibility

### Data
Our in house dataset used to train and validate the model is not available, however, the external validation datasets are accessible. Both CAMELYON datasets can be downloaded from https://camelyon17.grand-challenge.org/Data/, TCIA-DLBCL from https://www.cancerimagingarchive.net/collection/dlbcl-morphology/, TCIA-SLN-Breast from https://www.cancerimagingarchive.net/collection/sln-breast/, and IICBU from https://andrewjanowczyk.com/deep-learning/

### Model parameters
Model hyperparameters for each MIL pooler and feature extractor are saved in ````trained_models/``` in json format.

## License and citation
If you find xMIL-LRP to be useful for your work, please cite the original article
```
@inproceedings{hense2024xmil,
  author = {Hense, Julius and Jamshidi Idaji, Mina and Eberle, Oliver and Schnake, Thomas and Dippel, Jonas and Ciernik, Laure and Buchstab, Oliver and Mock, Andreas and Klauschen, Frederick and M\"{u}ller, Klaus-Robert},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {8300--8328},
  publisher = {Curran Associates, Inc.},
  title = {xMIL: Insightful Explanations for Multiple Instance Learning in Histopathology},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/0f9e0309d8a947ca44463a9b7e8b6a3f-Paper-Conference.pdf},
  volume = {37},
  year = {2024}
}
```

:copyright: This code is provided under the MIT License. Please refer to the license file for details.

Note: the license was updated from CC BY-NC-ND 4.0 in April 2026.
