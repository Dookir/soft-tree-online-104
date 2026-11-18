# Data and artifact provenance

- `adaptive/data/train.csv` is the original 2,048-row course-provided offline dataset, included here as part of the author's requested teaching/reproduction repository. Its origin is the course; this repository does not claim it was newly collected by the model authors.
- `results/formal-v4/online.json` and the20 input/response batches contain the1,000 scalar feedback labels generated in this independent experiment using the recovered evaluator. These labels are additional online feedback, not original offline training data.
- `results/formal-v4/models/v00/` contains five models freshly trained only on the original dataset. No old pretrained model or online feedback initializes v4. The final model `results/formal-v4/models/v20/` was trained with this experiment's1,000 feedback rows as well. The complete v1 package remains at tags v1.0.0 and v3.0.0.
- The MIT license applies to this repository's implementation code. It does not purport to replace any data provider's terms or third-party rights in course data, original evaluator software, runtime assets, or dependencies.
- The archived v4.0 recovered-backend experiment does not distribute recovered bytecode, evaluator runtime inputs, or third-party temperature models. Supply a compatible JSON backend to obtain new true evaluations. Historical input/R pairs can be checked, and model training can be replayed, without that backend.

- `results/formal-v4/verification/best-net-traces.json` contains only the archived best case’s8-week district/building net-load outputs, enabling independent recomputation of the reported scalar R without redistributing evaluator runtime inputs.

- The Windows CPU v4.1.0 archive is in windows-cpu-a/evidence. Its model artifacts and 1,000 scalar feedback rows were produced with the original Windows scorer. The original executable is distributed separately as a v4.1.0 release asset.
