
I really have a bit of difficult in finding the correct version since nvidia documentation for this case is not very precise. So, considering that my cuda is 2.6, jetpack 6, then I should install to my Jetson Orin Nano the v61, which can be found in this website:

https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/

# 1. Download the specific wheel (nv24.07)
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl

# 2. Install it
pip3 install torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl




