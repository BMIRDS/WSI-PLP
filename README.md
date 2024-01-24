
## Quick Setup for MaskHIT
For a faster MaskHIT setup:
* Run `requirement.sh` to install needed packages

* Install MaskHIT with `setup_maskhit.py`


# WSI-PLP: Whole Slide Image Analysis for Patient-level Predictions
The WSI-PLP repository provides tools for analyzing Whole Slide Images (WSI) with a focus on making predictions at the patient level. It's built for handling large slide images in pathology.
This repository includes two weakly-supervised deep learning methods for digital pathology and whole slide image (WSI) analysis:

## 1. POPPSlide: Patient Outcome Prediction Pipeline using Whole Slide Images
1. Go to the `POPPSlide` folder.
2. See the README there for installation details.

POPPSlide offers a comprehensive pipeline for predicting patient outcomes (categorical, time to event, or continuous) using WSIs.

Detailed method description: https://www.nature.com/articles/s41598-021-95948-x.

## 2. MaskHIT: Masked Pre-Training of Transformers for Histology Image Analysis
1. Visit the `MaskHIT` folder.
2. The README explains setup and configurations.

MaskHIT utilizes a masked language model-like pretext task to train transformers on WSIs without labeled data.
- **Performance**: Outperforms various multiple instance learning approaches by 3% in survival prediction and 2% in cancer subtype classification tasks, and exceeds recent transformer-based methods.
- **Validation**: Attention maps generated align with pathologist annotations, indicating accurate identification of relevant histological structures.

For more information:
https://arxiv.org/abs/2304.07434

# Installation
1. **Dependencies**

* For installing necessary dependencies, use the provided script:
`install_requirements.sh`

* For Singularity/Docker environment, use:
`install_requirements_for_container.sh`

2. **Package Installation**

* Install this MaskHIT package with a setup script `python setup_maskhit.py install`.

# Usage
- **POPPSlide**

Navigate to the `POPPSlide` subfolder for details and instructions.

- **MaskHIT**

For utilizing the latest pipeline, refer to the `maskhit` subfolder.

* TODO: Rename maskhit folder
* Note: The maskhit folder will be renamed for consistency.
