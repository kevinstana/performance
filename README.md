dataset link: https://drive.google.com/file/d/10s8MCGoyq3_iXaKrSWLgaR3Knt9qQ3BT/view

Extract the contents in the directory you saved the .py files.
```bash
tar -xvzf 'dermoscopy_classification.tar.gz'
```

Download modules:
```bash
pip install psutil
pip install numpy
pip install pandas
pip install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Run
```bash
python3 performance_inference.py
```
```bash
python3 performance_training.py
```
If python3 doesn't work, try python
