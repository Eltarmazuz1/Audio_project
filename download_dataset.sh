wget -q --show-progress https://github.com/karoldvl/ESC-50/archive/master.zip -O dataset/ESC-50.zip
unzip -q  dataset/ESC-50.zip -d  dataset/ESC-50
mv dataset/ESC-50/ESC-50-master/* dataset/ESC-50/ && rm -r dataset/ESC-50/ESC-50-master
