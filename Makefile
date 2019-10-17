# check for python installation
# null_string = 
# ifeq ($(null_string), $(which python))
#   $(error "PYTHON=$(PYTHON) installation not found")
# endif
# check python version
# PYV=$(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)");

requirements:
	pip install -r requirements.txt

setup: 
	python setup.py install
	python -m nltk.downloader brown stopwords

run:
	py main.py
