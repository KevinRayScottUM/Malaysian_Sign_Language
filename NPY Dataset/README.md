# Processed NPY Dataset

The processed `.npy` arrays are generated artifacts and are not committed because `X_TRAIN_2.npy` is over GitHub's 100 MiB file limit.

To recreate the files, download `BIM Dataset V3` first, then run the split/build scripts from the project root:

```bash
python3 Dataset_split_6_2_2.py
python3 build_npy_from_split.py
```

The repository keeps lightweight metadata such as `label_map.json` and path lists when available.
