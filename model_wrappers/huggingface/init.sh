#!/bin/bash

if $INSTALL_DEP; then
    # transformers pinned to keep the already-verified ChemBERTa/ChemGPT/ChemFM
    # variants stable -- MolFormer needs transformers>=5.0.0 instead (see
    # model_wrappers/huggingface/venv_molformer/), no single version supports both.
    pip install "transformers[torch]==4.46.3" selfies==2.1.1 rdkit

    pip install sentencepiece protobuf sentence-transformers

    echo $(pwd)

    if [ ! -f ./SELFormer.zip ] && [ ! -d ./model_weights/SELFormer ]; then
        echo "[!!!] For SELFORMER to work, you need to manually download the model weights from the following link and place it in the current directory"
        echo "Download link -> https://drive.google.com/drive/folders/1c3Mwc3j4M0PHk_iORrKU_V5cuxkD9aM6"
        # exit 10
    fi

    if [ -f ./SELFormer.zip ]; then
        mkdir -p ./model_weights/SELFormer
        unzip SELFormer.zip -d ./model_weights/SELFormer-tmp
        mv ./model_weights/SELFormer-tmp/SELFormer/* ./model_weights/SELFormer
        rm SELFormer.zip && rm -r ./model_weights/SELFormer-tmp
    fi

    if [ -f ./SELFormer-Lite.zip ]; then
        mkdir -p ./model_weights/SELFormer-Lite
        unzip SELFormer-Lite.zip -d ./model_weights/SELFormer-Lite-tmp
        mv ./model_weights/SELFormer-Lite-tmp/SELFormer-Lite/* ./model_weights/SELFormer-Lite
        rm SELFormer-Lite.zip && rm -r ./model_weights/SELFormer-Lite-tmp
    fi
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HYDRA_EXPERIMENT=huggingface
