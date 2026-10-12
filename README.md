# Vegetable Classification with Lightweight CNNs

A image classifier that recognises 15 kinds of vegetables from low-resolution 64×64 images, built with the kind of hardware you'd find on an edge device in mind: a Raspberry Pi, a microcontroller, or a cheap camera node.

We compare two CNNs trained from scratch. One is a standard convolutional network. The other is a lightweight version built on depthwise-separable convolutions and kept under 100k parameters. We also fine-tune two pre-trained mobile architectures (MobileNetV2 and EfficientNet-B0) and look at how they all trade off accuracy, memory footprint and compute.

## Dataset

We use the [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) from Kaggle. It has 21,000 images across 15 classes: bean, bitter gourd, bottle gourd, brinjal, broccoli, cabbage, capsicum, carrot, cauliflower, cucumber, papaya, potato, pumpkin, radish and tomato.

The original 224×224 images are downscaled to 64×64 to mimic a low-resolution sensor. We keep the dataset's own train/validation/test split and remove a few exact duplicates. That leaves **14,996 / 3,000 / 3,000** images (about 71/14/14%). The split is fixed in `splits.csv`, so everyone working on the project trains and tests on exactly the same images.

## Project layout

```
├── scripts/
│   ├── download_data.py    # fetches the dataset into data/raw
│   └── prepare_data.py     # builds the 64×64 dataset in data/processed from splits.csv
├── src/                    # shared code: data loading, models, training and evaluation
├── notebooks/              # experiments and results
├── results/                # figures and metrics
├── splits.csv              # the fixed train/val/test split
└── requirements.txt
```

`data/` and `models/` are generated locally and aren't tracked in git.

## Getting started

The project uses Python 3.13 and TensorFlow 2.21 (with Keras 3), plus scikit-learn, pandas, Pillow, matplotlib/seaborn and kagglehub. Exact versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py
python scripts/prepare_data.py
```

`prepare_data.py` checks every image against the hashes in `splits.csv` and should finish with 14,996 / 3,000 / 3,000 images. On Windows, clone into a short path such as `D:\Projects\vegCNN` (or enable long paths), because TensorFlow's install fails on deeply nested folders.

TensorFlow runs on the CPU on native Windows. At 64×64 that's perfectly workable.

## Contributors

- Thevinu Fernando ([@thevinufernando](https://github.com/thevinufernando))
- Januka Sanujaya
- Thisura Samuditha ([@ThisuraSamuditha](https://github.com/ThisuraSamuditha))
- Chamod Dharmakeerthi ([@ChamodLD](https://github.com/ChamodLD))

## References

1. M. I. Ahmed, S. M. Mamun and A. U. Z. Asif, "DCNN-Based Vegetable Image Classification Using Transfer Learning: A Comparative Study," *5th International Conference on Computer, Communication and Signal Processing (ICCCSP)*, 2021.
2. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-Based Learning Applied to Document Recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998.
3. F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," *CVPR*, 2017.
4. A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017.
5. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *CVPR*, 2018.
6. M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *ICML*, 2019.
7. K. P. Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.

## License

Released under the [MIT License](LICENSE).
