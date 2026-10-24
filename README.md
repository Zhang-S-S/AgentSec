# AgentSec

AgentSec is an account-level risk detector for agent social platforms. It combines
Residual Top-k multiple-instance learning over posts and comments, auxiliary
profile and URL evidence, and directed message passing over the comment graph.

## Repository layout

```text
comment_network/   directed graph construction and graph encoders
multisource/       account encoders and Residual Top-k MIL
offline_8b/        Qwen3-Embedding-8B preprocessing and embedding pipeline
agentsec/              deterministic five-fold experiment orchestration
scripts/           command-line entry points
configs/           paper configuration
tests/             lightweight configuration tests
```

Datasets, labels, embeddings, model checkpoints, and experimental outputs are
not included. Keep them outside the repository and provide their paths through
the command line.

## Environment

Python 3.10 or newer and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m pip install -e .
python -m pip install -r offline_8b/requirements.txt
```

## Qwen3-Embedding-8B features

The embedding pipeline is offline and records the model revision, model hash,
preprocessing configuration, split membership, and source fingerprints.

```bash
python -m offline_8b.prepare \
  --source-db /path/to/platform.db \
  --label-db /path/to/labels.db \
  --splits /path/to/splits.json \
  --output-dir /path/to/prepared \
  --resume

python -m offline_8b.encode \
  --input-dir /path/to/prepared \
  --output-dir /path/to/embeddings \
  --model-path /path/to/Qwen3-Embedding-8B \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch-size 8 \
  --max-length 2048 \
  --profile-max-length 512 \
  --embedding-dim 4096 \
  --instruction \
  --resume

python -m offline_8b.verify \
  --input-dir /path/to/embeddings \
  --prepared-dir /path/to/prepared
```

## Directed comment graph

```bash
python scripts/build_graph.py \
  --source-db /path/to/platform.db \
  --labels /path/to/labels.db \
  --output-dir /path/to/graph \
  --labeled-only
```

The training command expects `graph_arrays.npz` produced by this stage.

### Directional mixing analysis

The following command compares Risky-to-Normal and Normal-to-Risky edges,
comment-weighted interaction intensity, reciprocal cross-label pairs, and
significance under account-level bootstrap and label permutation:

```bash
python scripts/analyze_directional_mixing.py \
  --graph-arrays /path/to/graph/graph_arrays.npz \
  --output-dir /path/to/outputs/directional_mixing \
  --permutations 10000 \
  --bootstrap 10000 \
  --seed 42
```

## Five-fold evaluation

The paper configuration uses seed 42, post top-k 3, comment top-k 5, four
incoming and outgoing neighbors, and text-loss weight 1.0. Each fold selects
its decision threshold using only the validation partition.

```bash
python scripts/train_fivefold.py \
  --embedding-dir /path/to/embeddings \
  --graph-arrays /path/to/graph/graph_arrays.npz \
  --reference-splits-root /path/to/reference/folds \
  --output-root /path/to/outputs/full \
  --variant full \
  --device cuda:0 \
  --num-workers 4
```

Omit `--reference-splits-root` to create deterministic stratified group folds.
Use the same reference folds for every method in a comparison.

## Ablations and sensitivity analysis

```bash
python scripts/run_ablations.py \
  --embedding-dir /path/to/embeddings \
  --graph-arrays /path/to/graph/graph_arrays.npz \
  --reference-splits-root /path/to/reference/folds \
  --output-root /path/to/outputs/ablations
```

The ablation suite covers mean pooling without Residual Top-k MIL, pure
attention MIL, removal of the graph branch, removal of edge directionality,
removal of auxiliary information, and removal of text supervision.

The generic sensitivity runner accepts comma-separated grids. For example:

```bash
python scripts/run_sensitivity.py \
  --embedding-dir /path/to/embeddings \
  --graph-arrays /path/to/graph/graph_arrays.npz \
  --reference-splits-root /path/to/reference/folds \
  --output-root /path/to/outputs/topk \
  --post-top-k 1,3,5,7,9 \
  --comment-top-k 1,5,10,15,25
```

Every run stores its configuration, input hashes, fold-level predictions,
per-fold metrics, pooled out-of-fold metrics, and mean and standard deviation.

## Temporal deployment simulation

The temporal runner trains on May 1--17 using an account-level train/validation
split, scores accounts with cumulative information available through May 24,
and retrospectively evaluates direct risky-to-normal interactions from May 25
through June 1. Post, comment, URL, and graph inputs are truncated at each
cutoff. Static profile snapshots are retained and explicitly recorded as such.

```bash
python scripts/run_temporal_intervention.py \
  --source-db /path/to/platform.db \
  --labels-db /path/to/labels.db \
  --embedding-dir /path/to/embeddings \
  --output-root /path/to/outputs/temporal_intervention \
  --device cuda:0 \
  --num-workers 4 \
  --seed 42
```

The output includes validation-selected predictions, the risk ranking, direct
exposure counts, and matched-budget comparisons against random selection and
historical out-degree. The simulation removes observed outgoing interactions;
it does not estimate downstream causal behavior changes.

## Reproducibility

The entry point enables deterministic PyTorch algorithms and deterministic
CUDA settings. URL byte-CNN pooling uses an explicit maximum reduction. The
same seed, fold files, embeddings, graph arrays, and software environment are
required for an exact reproduction.

## Tests

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s offline_8b/tests -v
```
