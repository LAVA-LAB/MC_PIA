# Install graphviy
sudo apt-get install graphviz -y
mkdir engine
cd engine
# Create Virtual-Envrionment
python -m venv env
source env/bin/activate
pip install -r requirements.txt
# Install PyCarl
git clone https://github.com/moves-rwth/pycarl.git
cd pycarl
python setup.py build_ext --jobs 1 develop
cd ..
# Install Stormpy
git clone https://github.com/moves-rwth/stormpy.git
cd stormpy
python setup.py build_ext --jobs 1 develop