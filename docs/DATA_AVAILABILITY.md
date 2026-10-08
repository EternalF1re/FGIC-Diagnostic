# Data and result availability

The original datasets are obtained separately from their providers:

| Dataset | Provider | Development pool used for OOF |
|---|---|---|
| CUB-200-2011 | [Caltech/UCSD project page](https://www.vision.caltech.edu/datasets/cub_200_2011/) | Official training partition, 5,994 samples, 200 classes |
| Stanford Cars | [Original Stanford project](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | Official training partition, 8,144 samples, 196 classes |
| Oxford Flowers-102 | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | Official train plus validation, 2,040 samples, 102 classes |
| Classify Leaves | [Kaggle competition](https://www.kaggle.com/competitions/classify-leaves) | Labeled training CSV, 18,353 samples, 176 classes |

Provider terms govern images and dataset redistribution. The CUB provider explicitly restricts image use to non-commercial research and education. A code MIT license does not grant rights over dataset images or third-party weights. Download availability and authentication can change; consult the provider terms. No images or third-party weights are bundled here.

Published CSVs are author-generated historical predictions (sample identifiers, fold, ground truth and predicted class) and recorded research statistics. Sample IDs are dataset identifiers, not personal identifiers. Numeric Leaves IDs refer to the original labeled-CSV row order. External sample IDs preserve the original dataset identifier. Class labels are the recorded zero-based label mapping; saved class-map records are provided where retained.

OOF development pools never include the provider's official test split. Secondary official-test summary records, when present in copied logs, are not substituted for OOF. See `DATASETS.md` for preparing new manifests and `RESULTS_PROVENANCE.md` for historical artifacts.
