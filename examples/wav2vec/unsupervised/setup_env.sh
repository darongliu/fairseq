# create conda 
# because there will be bug when importing torch in lab's computer, therefore do not use conda
# conda create -n wav2vecu python=3.8 -y
# conda activate wav2vecu

# pip install fairseq
cd ../../../.. # to fairseq root
pip install --editable ./ # install --user

pip install soundfile scipy packaging editdistance npy-append-array g2p_en  # install --user

# install KenLM # run at unsupervised dir
git clone https://github.com/kpu/kenlm
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 4
cd ..
python setup.py install
cd ..

# download wave2vec model
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt

# intstall faiss
pip install faiss-gpu  # install --user

# install flashlight so that can run decoding
git clone --branch v0.3.2 https://github.com/flashlight/flashlight
cd flashlight/bindings/python/
# modify the cmakefile # delete #130, 22-24 in flashlight/cmake/FindFFTW3.cmake
KENLM_ROOT=/home/darong/darong/fairseq/examples/wav2vec/unsupervised/kenlm \
python setup.py install --user

# set env rariable
export FAIRSEQ_ROOT=/home/darong/darong/fairseq
export KALDI_ROOT=/opt/kaldi
export KENLM_ROOT=/home/darong/darong/fairseq/examples/wav2vec/unsupervised/kenlm/build/bin
# init LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:/lib
