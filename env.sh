#conda create -n aspire python=3.9;conda activate aspire;bash env.sh

pip install freerec
conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install pandas tqdm matplotlib torchdata=0.7.1 pyg -c pyg
pip install scikit-learn einops