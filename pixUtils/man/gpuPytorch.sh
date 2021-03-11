eval "$(conda shell.bash hook)"
pkgName=gputf
if [ "$pkgName" == "" ]
then
  echo "Please specify package name"
  exit
fi

conda create -n $pkgName --clone base2 $2
conda activate $pkgName


condaArgs=""
pipArgs=""


#pipArgs+=" tensorflow-cpu==1.15.0"
#condaArgs+=" tensorflow-gpu==1.15.0"

#condaArgs+=" jupyterlab"
condaArgs+=" ffmpeg==4.1.3"

condaArgs+=" "

#pipArgs+=" torch==1.2.0"
pipArgs+=" torch==1.7.1"
pipArgs+=" redis==3.5.3"
pipArgs+=" tqdm==4.47.0"
pipArgs+=" aiofiles==0.5.0"
pipArgs+=" Pillow==7.1.2"
pipArgs+=" boto3==1.14.34"
pipArgs+=" uvicorn==0.11.8"
pipArgs+=" fastapi==0.60.1"
pipArgs+=" torchvision==0.8.2"
pipArgs+=" albumentations==0.5.2"
pipArgs+=" python-multipart==0.0.5"
pipArgs+=" opencv-python==4.1.0.25"
# pipArgs+=" tensorflow-serving-api==1.15.0"
pipArgs+=" "


echo ___________________________________________________________________
echo conda install -c conda-forge -y -----${condaArgs}
conda install -c conda-forge -y ${condaArgs}

echo ___________________________________________________________________
echo pip install -------------------------${pipArgs}
pip install ${pipArgs}
#pip install --upgrade jupyter_http_over_ws>=0.0.7 && jupyter serverextension enable --py jupyter_http_over_ws

if [[ "$USER" == "ec2-user" ]]
then
  # clear unwanted stuff
  sudo yum autoremove;
  sudo yum autoclean;
  sudo yum clean expire-cache
  conda clean --all;
  rm -rf ~/.cache/pip
fi