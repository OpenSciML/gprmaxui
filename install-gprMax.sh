git clone https://github.com/gprMax/gprMax.git
cd  gprMax
git pull
python setup.py cleanall
python setup.py build
python setup.py develop --no-deps