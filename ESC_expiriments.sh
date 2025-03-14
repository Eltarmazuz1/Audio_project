wget -q --show-progress https://github.com/karoldvl/ESC-50/archive/master.zip -O dataset/ESC-50.zip
unzip -q  dataset/ESC-50.zip -d  dataset/ESC-50
mv dataset/ESC-50/ESC-50-master/* dataset/ESC-50/ && rm -r dataset/ESC-50/ESC-50-master

# Conformer Adapter Pfeiffer
python3 main.py --data_path dataset/ --is_AST True --dataset_name 'ESC-50' --method 'adapter' --seq_or_par 'parallel' --reduction_rate_adapter 64 --adapter_type 'Pfeiffer' --apply_residual False --adapter_block 'conformer'

# Linear
python3 main.py --data_path dataset/ --is_AST True --dataset_name 'ESC-50' --method 'linear' 