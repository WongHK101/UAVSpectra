# UAVSpectra

UAV RGB-multispectral reconstruction with GCP, LiDAR, image-derived geometry,
appearance and spectral-product evaluation. UMGS is the reference baseline.

**Dataset:** [Baidu Netdisk](https://pan.baidu.com/s/11o_eXMEXjP4yH16395O5nQ?pwd=fcif) (access code: `fcif`).

Under `UAVSpectra-dataset/`, `all21scenes/` groups the 21 scene image folders,
17 OpenMVS proxy meshes, and airborne LiDAR for all six Urban scenes.
`all21scenes/GCP/` contains their manual image annotations, surveyed coordinates
and control/checkpoint assignments. `6scenesinpaper/`
contains the six-scene comparison used in the paper. Urban50K retains one extra
raw capture (0001); use the supplied 2,208-view manifest for that scene.

## Layout

- `baseline/`: image preparation, training, rendering and CUDA extensions.
- `evaluation/`: scorers, method exporters and fixed evaluation settings.
- `evaluate.py`: GCP, LiDAR and proxy evaluation entry point.
- `unpack_data.py`: extract the accompanying scene downloads.

The six-scene comparison package contains the RGB and narrowband captures, prepared inputs,
camera records, fixed splits and proxy depths. Urban3K and Urban5K also include
the manual image annotations, surveyed coordinates and airborne LiDAR used for
scoring. Download the ZIPs from `6scenesinpaper/` into `../UAVSpectra-dataset/`,
then extract them:

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
