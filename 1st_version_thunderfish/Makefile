INSTALLDIR=/opt/thunderfish
LINKNAME=/usr/bin/thunderfish

python-odml:
	git clone https://github.com/G-Node/python-odml.git

install: python-odml
	chmod a+rx ./thunderFish.py
	if [ ! -d $(INSTALLDIR) ]; then sudo mkdir $(INSTALLDIR); fi
	sudo rsync -av * $(INSTALLDIR)
	sudo ln -s $(INSTALLDIR)/thunderFish.py $(LINKNAME)

update: python-odml
	git pull origin master
	cd python-odml && git pull origin master
	echo "\n\nIf you want to install it system wide you need to execute make install now!\n\n"

clean:
	rm -rf python-odml

uninstall:
	sudo rm -rf $(INSTALLDIR)
	sudo rm $(LINKNAME)
