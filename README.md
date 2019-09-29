# precalc-fid-pytorch
Ported existing code to use pytorch datasets

# Sample command line use:
This command downsamples FFHQ dataset from 1024x1024 to 256x256, normalizes and precalculates its statistics:

`python precalc_fid.py --dataroot ../ffhq-dataset/images1024x1024/ --imageSize 256 --outf fid_stats_ffhq256.py --normalize --workers 4`

# Acknowledgement
Code is based on: https://github.com/mseitzer/pytorch-fid
