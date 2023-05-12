conda init
eval "$(conda shell.bash hook)"
conda activate tsym
python train.py
wait
python learnpixel2prop.py

