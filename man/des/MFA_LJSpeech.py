

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321markdown
'''
### Forced align LJSpeech dataset using Montreal Forced Aligner (MFA)


**Note**: The notebook takes ~2 hours to finish.

Expected results:

<img src="https://i.imgur.com/5uehkba.png"></img>

'''

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# %%writefile install_mfa.sh
# #!/bin/bash

## a script to install Montreal Forced Aligner (MFA)

root_dir=${1:-/tmp/mfa}
mkdir -p $root_dir
cd $root_dir

# download miniconda3
wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $root_dir/miniconda3 -f

# create py38 env
$root_dir/miniconda3/bin/conda create -n aligner -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch -y
source $root_dir/miniconda3/bin/activate aligner

# install mfa, download kaldi
pip install montreal-forced-aligner # install requirements
pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git # install latest updates

mfa thirdparty download

echo -e "\n======== DONE =========="
echo -e "\nTo activate MFA, run: source $root_dir/miniconda3/bin/activate aligner"
echo -e "\nTo delete MFA, run: rm -rf $root_dir"
echo -e "\nSee: https://montreal-forced-aligner.readthedocs.io/en/latest/aligning.html to know how to use MFA"

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# download and install mfa
INSTALL_DIR="/tmp/mfa" # path to install directory

# !bash ./install_mfa.sh {INSTALL_DIR}
# !source {INSTALL_DIR}/miniconda3/bin/activate aligner; mfa align --help

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# download and unpack ljs dataset
# !echo "download and unpack ljs dataset"
# !mkdir -p ./ljs; cd ./ljs; wget -q --show-progress https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# !cd ./ljs; tar xjf LJSpeech-1.1.tar.bz2

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# install sox tool
# !sudo apt install -q -y sox
# convert to 16k audio clips
# !mkdir ./wav
# !echo "normalize audio clips to sample rate of 16k"
# !find ./ljs -name "*.wav" -type f -execdir sox --norm=-3 {} -r 16k -c 1 `pwd`/wav/{} \;
# !echo "Number of clips" $(ls ./wav/ | wc -l)

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# create transcript files from metadata.csv
lines = open('./ljs/LJSpeech-1.1/metadata.csv', 'r').readlines()
from tqdm.auto import tqdm
for line in tqdm(lines):
  fn, _, transcript = line.strip().split('|')
  ident = fn
  open(f'./wav/{ident}.txt', 'w').write(transcript)

# this is an example transcript for LJ001-0001.wav
# !cat ./wav/LJ001-0001.txt

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# download a pretrained english acoustic model, and english lexicon
# !wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip
# !wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
# FINALLY, align phonemes and speech
# !source {INSTALL_DIR}/miniconda3/bin/activate aligner; \
mfa align -t ./temp -c -j 4 ./wav librispeech-lexicon.txt ./english.zip ./ljs_aligned
# output files are at ./ljs_aligned
# !echo "See output files at ./ljs_aligned"

# =============================================================================================================================================sdfgver%$^#$^&*%^fd321code
