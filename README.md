

# ASPIRE

PyTorch implementation of "ASPIRE: Make Spectral Graph Collaborative Filtering Great Again via Adaptive Filter Learning"

## Environment Setup

    conda create -n aspire python=3.9;conda activate aspire;bash env.sh
    unzip datasets.zip -d data/Processed
    cd ASPIRE

If you encounter errors when running the code, please try:

    pip install numpy==1.26.4

## Usage

### Homogeneous Setting

    python main.py --config=configs/Amazon2014Baby_550_MMRec.yaml --root=../data

### Heterogeneous Setting

    python main.py --config=configs/Amazon2014Baby_550_MMRec_heterogeneous.yaml --root=../data

### Dual-Branch Setting

    python main.py --config=configs/Amazon2014Baby_550_MMRec_dualbranch.yaml --root=../data

## Citation

If you find STAIR useful in your research, please kindly cite:

    @misc{he2026aspire,
        title={ASPIRE: Make Spectral Graph Collaborative Filtering Great Again via Adaptive Filter Learning}, 
        author={Yunhang He and Cong Xu and Zhangchi Zhu and Hongzhi Yin and Wei Zhang},
        year={2026},
        eprint={2604.22549},
        archivePrefix={arXiv},
        primaryClass={cs.IR},
        url={https://arxiv.org/abs/2604.22549}, 
    }