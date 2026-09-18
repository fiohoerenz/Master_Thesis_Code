# ki-sort-master-thesis

Multimodal plastic packaging sorting and recyclability assessment pipeline —
YOLO-based instance segmentation (body/cap/label), NIR/MIR spectral polymer
classification, and an EN 18120-grounded impurity index. Developed as part of
a master's thesis at Hochschule Aalen.

The dataset behind this pipeline — **InSPACK (Instance-segmented Spectral
Packaging)**, a component-level (body/cap/label) multimodal RGB + FT-NIR
dataset — is published separately: **[InSPACK](<link once that repo exists>)**
(also archived on Zenodo: `<DOI once minted>`).

## Repository structure

```
ki-sort-master-thesis/
├── README.md, LICENSE, CITATION.cff
├── environment/          per-conda-env yml exports (yolo-train, ir-class, ...)
├── weights/              evaluation-tier and deployment-tier model weights only
│   └── README.md         quick reference table, see "Model weight tiers" below
└── code/                 one flat folder — notebooks/scripts are numbered
                          in execution order
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

Each filename starts with its pipeline stage (`00`–`08`) plus a per-stage
sequence number giving execution order within that stage (both zero-padded
to two digits, four digits total) — e.g. `05` = IR classification, and
`0501`–`0505` run in that order. No per-stage subfolders and no date
prefixes. `08_Generalizability` code only — external datasets (SpectralWaste,
WaRP, PackWISE) referenced there are NOT redistributed; you'll need to obtain
them yourself.

## Model weight tiers — do not conflate

KI-Sort maintains two tiers of trained model per modality (YOLO segmentation,
IR/NIR logistic-regression classifier). Keeping them straight matters:
mixing them up silently turns an honest out-of-sample metric into a
resubstitution one. (The thesis covers the statistical justification for this
split in full; this is just the practical summary needed to use the weights
correctly.)

- **Evaluation tier** — `yolov8n-seg-eval.pt`,
  `logistic-regression-eveluation.joblib` — fit on the cross-validation pool
  only (never touches held-out test data). Use this tier for every number
  that appears in the thesis as a performance metric.
- **Deployment tier** — `yolov8n-seg-deployment.pt`,
  `logistic-regression-deployment.joblib` — fit on all available data, to
  squeeze out the best possible model for actual use. Because it has seen
  everything, any metric computed by re-running it against its own training
  data is a resubstitution estimate — it will look better than the model
  actually is out of sample. Use this tier only for inference, demonstration,
  or predicted-label export where you specifically want the best available
  model rather than an honest metric.

If you retrain either model, keep this same two-tier structure and naming
convention so the distinction stays legible to anyone reading the repo later.

See `weights/README.md` for the same information as a quick-reference table.

## Setup

Each pipeline stage was run in a specific conda environment (`yolo-train`,
`ir-class`, and others) — see `environment/*.yml` for exact package versions.

Notebooks and scripts that read data reference a `KI_SORT_ROOT` environment
variable (defaulting to the original server path, `/VOL_D/fio.h`) rather than
a hardcoded path — set it to wherever you've placed the data before running
anything:
```
export KI_SORT_ROOT=/path/to/your/data
```

## External datasets used for generalizability evaluation

Stage `08_Generalizability` includes preprocessing and benchmarking *code*
for SpectralWaste, WaRP, and PackWISE, used to evaluate KI-Sort's models on
external data. None of that external data (raw or derived) is redistributed
here — obtain it from its original source if you want to reproduce those
benchmarks.

## Citation

If you use this code, please cite it — see `CITATION.cff`. If you also use
the InSPACK dataset, please cite it separately via its own DOI (see the
InSPACK repo).

## License

MIT — see `LICENSE`.
