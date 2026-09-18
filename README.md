# ki-sort-master-thesis

This repository contains the pipeline code and the trained model weights of
the master's thesis *KI-Sort: AI-Driven Vision and Infrared Sorting for Base
Material Identification and Impurity Indexing of Labeled Plastic Packaging
Waste* (Hochschule Aalen). The pipeline combines YOLO-based instance
segmentation of packaging components (body, cap, label), FT-NIR
diffuse-reflectance polymer classification, and an impurity index grounded in
EN 18120.

The dataset used by this pipeline, **InSPACK (Instance-segmented Spectral
Packaging)**, a component-level multimodal RGB and FT-NIR dataset, is
published separately: `<link to the InSPACK record / its DOI>`.

## Repository structure

```
ki-sort-master-thesis/
├── README.md, LICENSE, CITATION.cff
├── env/                  conda environment exports (see Setup)
├── weights/              evaluation-tier and deployment-tier model weights
└── code/                 one flat folder, numbered in execution order
    ├── 0001_Image_Preprocessing.ipynb
    ├── 0101_LS_Export.ipynb
    ├── 0201_Image_Segmentation_Model_Screening.ipynb
    ├── 0202_Image_Segmentation_Hyperparameter_Tuning_yolo26n.ipynb
    ├── 0203_Image_Segmentation_Hyperparameter_Tuning_yolov8n.ipynb
    ├── 0204_Image_Segmentation_Final_Training_yolo26n.ipynb
    ├── 0205_Image_Segmentation_Final_Training_yolov8n.ipynb
    ├── 0206_Image_Segmentation_Inference_Visualization.ipynb
    ├── 0207_Image_Segmentation_Deployment_Training.ipynb
    ├── 0301_Measurement_Point_Sampling.ipynb
    ├── 0401_IR_Preprocessing.ipynb
    ├── 0402_IR_Preprocessing_Data_Visualization.ipynb
    ├── 0501_IR_Classification_Benchmark_Preprocessing_Influence.ipynb
    ├── 0502_IR_Classification_Speed_Benchmark.ipynb
    ├── 0503_IR_Classification_Benchmark_Best_Preprocessing.ipynb
    ├── 0504_IR_Classification_LR_Deployment_Training.ipynb
    ├── 0505_IR_Classification_SVM_Deployment_Training.ipynb
    ├── 0601_Multimodal_Pipeline_Metadata_Correlation_Analysis.ipynb
    ├── 0602_Multimodal_Pipeline_Whole_Pipeline_Analysis.ipynb
    ├── 0701_Impurity_Index_Computation.ipynb
    ├── 0801_Generalizability_PackWISE_Preprocessing.ipynb
    ├── 0802_Generalizability_Spectral_Waste_Preprocessing.ipynb
    ├── 0803_Generalizability_WaRP_Preprocessing.ipynb
    ├── 0804_Generalizability_ls_export_truegt.py
    ├── 0805_Generalizability_ls_import_pseudogt.py
    ├── 0806_Generalizability_YOLO_Benchmark.ipynb
    ├── 0807_Generalizability_True_GT_Body_Valid_Export.py
    ├── 0808_Generalizability_Classifier_Assessment_VLIZ.ipynb
    ├── 0809_Generalizability_Classifier_Assessment_independent_capture_session.ipynb
    ├── 0810_Generalizability_YOLO_Finetune_PoC.ipynb
    └── 0811_Generalizability_Spectral_Waste_HSI_Investigation.ipynb
```

Each file name starts with its pipeline stage (`00` to `08`), followed by a
per-stage sequence number that states the execution order within that stage.
Both parts are zero-padded to two digits, resulting in four digits in total.
Stage `05`, for example, covers the NIR classification, and the files `0501`
to `0505` are executed in that order.

## Model weight tiers

Two tiers of trained model are maintained per modality, one for the YOLO
segmentation and one for the NIR classifier. The distinction is to be
observed, since an evaluation-tier metric and a deployment-tier metric are
not comparable. The statistical justification of the split is given in the
thesis. The summary required for using the weights is stated in the
following.

- **Evaluation tier** (`yolov8n-seg-eval.pt`,
  `logistic-regression-evaluation.joblib`): fit on the cross-validation pool
  only, without access to the held-out test objects. This tier is to be used
  for every performance metric reported in the thesis.
- **Deployment tier** (`yolov8n-seg-deployment.pt`,
  `logistic-regression-deployment.joblib`): fit on all available data. Any
  metric obtained by re-applying these models to their own training data
  constitutes a resubstitution estimate rather than an out-of-sample
  estimate. This tier is to be used for inference, demonstration, and
  predicted-label export only.

The authors recommend retaining the two-tier structure and the naming
convention if either model is retrained.

## Setup

Each pipeline stage was executed in a dedicated conda environment on the
training server (Ubuntu 22.04, CUDA 12.1). The folder `env/` holds one export
per environment, generated employing `conda env export --no-builds`, which
states the exact package versions:

| File | Environment | Used by |
|---|---|---|
| `env/scikit-image.yml` | `scikit-image` | `0001` |
| `env/ls-export.yml` | `ls-export` | `0101` |
| `env/yolo-train.yml` | `yolo-train` | `02xx`, `0301`, `06xx`, `0701`, `0806`, `0810` |
| `env/ir-class.yml` | `ir-class` | `04xx`, `05xx`, `0808`, `0809`, `0811` |
| `env/sam3-seg.yml` | `sam3-seg` | `0801` to `0803` |

An environment is recreated with:
```
conda env create -f env/yolo-train.yml
```
The exports represent the state of the environments at the time of export.
Individual package versions may therefore deviate from those employed for the
original runs.

Data paths are not hardcoded. Notebooks and scripts that read data resolve
the environment variable `KI_SORT_ROOT`, which defaults to the original
server path `/VOL_D/fio.h`. It is to be set to the location of the data
before execution:
```
export KI_SORT_ROOT=/path/to/your/data
```

## External datasets used for the generalizability assessment

Stage `08` contains preprocessing and benchmarking code for the external
datasets SpectralWaste, WaRP, and PackWISE, which were employed for the
assessment of the generalizability of the KI-Sort models. Neither the raw nor
the derived data of these datasets are redistributed in this repository. They
are to be obtained from their original sources in order to reproduce the
corresponding benchmarks.

## Note on AI assistance

The code in this repository was generated with the assistance of
Claude Sonnet 5 and Claude Opus 5 (Anthropic). The code was reviewed and
validated by the author.

## Citation

Citation information for this repository is provided in `CITATION.cff`. If
the InSPACK dataset is used as well, it is to be cited separately through its
own DOI.

## License

MIT, see `LICENSE`.
