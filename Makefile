project := gif-gan
flake8 := flake8

.PHONY: bootstrap
bootstrap:
	pip install -U "setuptools>=19,<20"
	pip install -U "pip>=7,<8"
	pip install -r requirements.txt
	
.PHONY: clean
clean:
	@find . "(" -name "*.pyc" ")" -delete

.PHONY: lint
lint:
	$(flake8) .

	
