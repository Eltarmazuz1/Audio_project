Audio Classification with AST and Fine-Tuning Methods
======================================================

------------------------------------------------------
SETUP INSTRUCTIONS
------------------------------------------------------

1. Install Required Packages:
-----------------------------
Make sure you have Python 3 installed. Then run:

    pip install -r requirements.txt

2. Download and Extract Dataset:
--------------------------------

(Option 1) ESC-50:
------------------
Download and extract the ESC-50 dataset:

    wget -q --show-progress https://github.com/karoldvl/ESC-50/archive/master.zip -O dataset/ESC-50.zip
    unzip -q dataset/ESC-50.zip -d dataset/ESC-50

(Option 2) UrbanSound8K:
-------------------------
Download and extract the UrbanSound8K dataset:

    wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1 -O dataset/UrbanSound8K.tar.gz
    tar -xzf dataset/UrbanSound8K.tar.gz -C dataset/

------------------------------------------------------
RUNNING EXAMPLES 
------------------------------------------------------

methods: 'linear', 'full-FT', 'adapter', 'prefix-tuning', 'LoRA', 'LoRA_freq'

1. Full Fine-Tuning on UrbanSound8K:

    python3 main.py --data_path dataset/ --is_AST True --dataset_name 'urbansound8k' --method 'full-FT'

2. LoRA Fine-Tuning on ESC-50:

    python3 main.py --data_path dataset/ --is_AST True --dataset_name 'ESC-50' --method 'LoRA'

3. Adapter Fine-Tuning (Parallel) on ESC-50 using Pfeiffer Adapter and Conformer Block:

    python3 main.py --data_path dataset/ --is_AST True --dataset_name 'ESC-50' --method 'adapter' --seq_or_par 'parallel' --adapter_type 'Pfeiffer' --adapter_block 'conformer'

------------------------------------------------------

