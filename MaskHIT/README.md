# Prepare the datasets to be used for this repo: see WSI_PreProcess
The following items should be presented in this repo:
* data folder: contains the preprocessed feature vectors
    - features of patches from one entire svs is stored as one .pt file, one row per one patch
        + this .pt file is stored in data/`<CANCER>`/`<id_svs>`/`<PATCH_SPEC>`/`<BACKBONE_SPEC>`/ subfolder
    - meta information for this .pt file is stored in data/`<CANCER>`/`<id_svs>`/`<PATCH_SPEC>`/ subfolder and in pickle format
        + the meta file is one row per patch, and contains at least the following colomns:
            * pos_x: x axis position of the patch on the WSI
            * pos_y: y axis position of the patch on the WSI
            * pos: [pos_x, pos_y]
            * valid: if this patch is background or not
            * counts_20: number of valid patches on a 20x20 tile [pos_x:pos_x+20, pos_y:pos_y+20]
    - 
* meta folder: contains the meta files
    - meta file for each individual `<META_PATIENT>`: id_patient, outcome, fold
    - meta file for each svs `<META_SVS>`: id_patient, id_svs, slide_type
* checkpoints folder: if not exist, will be created during training
    - checkpoints will be stored in checkpoints/`<TIMESTR>`/ subfolder
    - **unique `<TIMESTR>` is required to avoid checkpoints being overwritten**
* logs folder: if not exist, will be created during training
    - two logs will be generated for each experiment:
        + "`<TIMESTR>`\_meta.log": major logging file
        + "`<TIMESTR>`\_data.log": stores the training/validation results
    - log files will be stored in logs/`<STUDY>`/ subfolder

# Supported architectures for pooling patches features from regions
## MIL - average pooling
`--mil1=ap`

## MIL - attention
`--mil1=attn`

## MIL - multi-head attention
`--mil1=mhattn`

## MIL - vit
`--mil1=vit_h8l12`: vit with 8 **h**eads and 12 **l**ayers


# Data sampling

## sample svs from patients
* `--repeats-per-epoch`: how many times to select one patient during each epoch
* `--svs-per-patient`: how many times to select one patient during each epoch for evaluation

## sample regions from whole slide
* `--magnification`: magnification level
* `--region-size`: size of tile (i.e., region) under specified magnification level
* `--sampling-threshold`: minimum number of patches within the region for this region to be eligible for sampling
* `--regions-per-svs`: number of regions to select from one svs

## sample patches from region
* `--num-patches`: number of patches to sample from each region: if 0, then it will sample all the patches from each region

## Determine overlap between regions and perform systematic sampling
* `--grid-size`: specify the grid size when sampling regions from WSI. For example, for a region containing LxL patches, 
    - grid size = 1: sample regions with max overlap
    - grid size = L/2: sample regions with 50% overlap
    - grid size = L: no overlap
* Determine number of regions to sample from the WSI:
    - `python utils/get_region_info.py --meta-svs=meta/tcga_brca_svs.pickle --grid-size=10`
    - From the output, determine a reasonable number of regions to sample from each WSI. For example, we can use 64 regions if the majority of WSIs contain less than 64 regions.
* Specify the grid size and number of regions in model training. To save computational memory, specify the number of patches to sample from each region
    - `--grid-size=10 --regions-per-svs=64 --num-patches=100`

# Fine-tune for downstream tasks

## Example commands:

Single fold finetuning:
`python train.py --study=colon --cancer=TCGA_COAD --mil1=vit_h8l12 --mil2=ap --num-patches=0 --meta-svs=meta/tcga_coad_svs.pickle --meta-all=meta/tcga_coad_meta.pickle --lr-attn=1e-5 --lr-pred=1e-3 --wd=0.01 --ffpe-only --outcome=status --outcome-type=survival --sample-patient --dropout=0.2 -b=4 --resume=pretrained_20221013_201713 --resume-epoch=0500 --resume-fuzzy --fold=0 --timestr=20230120_120000`

5-fold cross-validation:
`python cross_validation.py colon 20230120_120000 --cancer=TCGA_COAD --mil1=vit_h8l12 --mil2=ap --num-patches=0 --meta-svs=meta/tcga_coad_svs.pickle --meta-all=meta/tcga_coad_meta.pickle --lr-attn=1e-5 --lr-pred=1e-3 --wd=0.01 --ffpe-only --outcome=status --outcome-type=survival --sample-patient --dropout=0.2 -b=4 --resume=pretrained_20221013_201713 --resume-epoch=0500 --resume-fuzzy`

Summary model performance:
`python utils/collect_predictions.py colon 20230120_120000`

Create attention maps
* Prepare a visualization dataset for selected slides
    - Modify and run `utils/prepare_visualization_data.py`
* Obtain attention values systematically across entire slide
    - Attention score for pre-trained model
        + `python train.py --cancer=TCGA_BRCA --mil1=vit_h8l12 --region-size=4480 --magnification=10 --ffpe-only --mode=extract --visualization --repeats-per-epoch=1 --regions-per-svs=1 --meta-svs=meta/vis_tcga_brca_locs-split.pickle --meta-all=meta/vis_tcga_brca_meta-split.pickle --resume=[pre-trained model name]    --resume-fuzzy --resume-epoch=0500 -b=64 --by-location --data=data`
    - Attention score for fine-tuned model
        + `python train.py --cancer=TCGA_BRCA --mil1=vit_h8l12 --region-size=4480 --magnification=10 --ffpe-only --mode=extract --visualization --repeats-per-epoch=1 --regions-per-svs=1 --meta-svs=meta/vis_tcga_brca_locs-split.pickle --meta-all=meta/vis_tcga_brca_meta-split.pickle --resume=[fine-tuned model name]  --resume-fuzzy --resume-epoch=BEST -b=64 --by-location --data=data`
* Create the attention map using Jupyter Notebook
    - `create attention maps.ipynb`

# Notations
* `<CANCER>`: name of cancer subset, such as TCGA_COAD, etc
* `<id_svs>`: svs file id
* `<PATCH_SPEC>`: format magnification_x-size_y, such as magnification_10-size_224
* `<BACKBONE_SPEC>`: name of backbone method for patch feature extraction, such as resnet_18
* `<STUDY>`: name of a study, such as ColonCancerSurvival
* `<TIMESTR>`: timestamp of an experiment, if none will auto generate timestamp