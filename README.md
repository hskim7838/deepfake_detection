## Quick Start (Optimized for T4 GPU (16GB))

'''bash
git clone https://github.com/hskim7838/deepfake_detection
cd deepfake_detection
conda create -n deepfake python=3.9 -y
conda activate deepfake
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

chmod +x train.sh
tmux new -s effort
./train.sh

