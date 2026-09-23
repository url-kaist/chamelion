install:
	@pip install -v .

install-all:
	@pip install -v ".[pseudo,viewer,test]"

uninstall:
	@pip -v uninstall chamelion

editable:
	@pip install -ve .
