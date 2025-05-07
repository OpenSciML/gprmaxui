sudo apt install libgomp1
sudo apt install libomp-dev


git clone https://github.com/gprMax/gprMax.git
cd gprMax || exit 1
python setup.py build
python setup.py install