# Data and artifact provenance

- `adaptive/data/train.csv` is the original 2,048-row course-provided offline dataset, included here as part of the author's requested teaching/reproduction repository. Its origin is the course; this repository does not claim it was newly collected by the model authors.
- `results/formal-104/online.json` and the20 input/response batches contain the1,000 scalar feedback labels generated in this independent experiment using the recovered evaluator. These labels are additional online feedback, not original offline training data.
- `pretrained_original/` contains five models trained only on the original dataset. The final model `results/formal-104/models/v20/` was trained with all1,000 feedback rows as well.
- The MIT license applies to this repository's implementation code. It does not purport to replace any data provider's terms or third-party rights in course data, original evaluator software, runtime assets, or dependencies.
- This repository does not redistribute the original closed executable, recovered bytecode, evaluator runtime inputs, or third-party temperature models. Supply a compatible JSON backend to obtain new true evaluations. Historical input/R pairs can be checked, and model training can be replayed, without that backend.
