# UAVSpectra

UAV RGB-multispectral reconstruction with GCP, LiDAR, image-derived geometry,
appearance and spectral-product evaluation. UMGS is the reference baseline.

**Dataset:** [UAVSpectra on Zenodo](https://doi.org/10.5281/zenodo.23930187).
The record is public; file access is currently restricted. Reviewers can use
the access link supplied with the manuscript.

## Layout

- `baseline/`: image preparation, training, rendering and CUDA extensions.
- `evaluation/`: scorers, method exporters and fixed evaluation settings.
- `evaluate.py`: GCP, LiDAR and proxy evaluation entry point.
- `unpack_data.py`: extract the accompanying scene downloads.

The six-scene dataset contains the RGB and narrowband captures, prepared inputs,
camera records, fixed splits and proxy depths. Urban3K and Urban5K also include
surveyed points and airborne LiDAR. Extract the scene downloads before scoring:

```bash
python unpack_data.py --data ../UAVSpectra-dataset
```

## Baseline

```bash
conda env create -f environment.yml
conda activate umgs
cd baseline
python -m pip install --no-build-isolation submodules/simple-knn
python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation submodules/fused-ssim
python run_spectralindexgs_pipeline.py --help
```

Use the supplied capture-group split and the paper's shared RGB SfM recipe.
Training and export recipes are in `evaluation/tools/measured_geometry_v1/training_recipes.py`.
Original captures and rectified training images are different image domains.

## Evaluation

In a separate CPU environment, install the evaluation dependencies. Given
a method-export directory containing `export_manifest.json` and its packets:

```bash
python -m pip install -r evaluation/requirements.txt
python evaluate.py --metric gcp --scene road --data ../UAVSpectra-dataset \
  --export-root /path/to/method_export --output /path/to/new_scores
```

See `evaluation/README.md` for LiDAR, appearance and proxy evaluation.
The entry point supplies the fixed reference settings. GCP/LiDAR retain the
original identity gates. The portable proxy interface compares declared
camera-matched packets; method/export qualification remains separate.

Third-party copyright and license terms are preserved in `LICENSE.md`,
`THIRD_PARTY_NOTICES.md` and the corresponding dependency directories.
