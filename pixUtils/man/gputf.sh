eval "$(conda shell.bash hook)"
pkgName=gputf
#pkgName=$1
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
# condaArgs+=" tensorflow-gpu==1.15.0"
condaArgs+=" tensorflow==2.3"




#condaArgs+=" jupyterlab"
condaArgs+=" ffmpeg==4.1.3"

condaArgs+=" "

pipArgs+=" redis==3.5.3"
pipArgs+=" tqdm==4.47.0"
pipArgs+=" aiofiles==0.5.0"
pipArgs+=" Pillow==7.1.2"
pipArgs+=" boto3==1.14.34"
pipArgs+=" uvicorn==0.11.8"
pipArgs+=" fastapi==0.60.1"
pipArgs+=" python-multipart==0.0.5"
pipArgs+=" opencv-python==4.1.0.25"
pipArgs+=" opencv-contrib-python==4.1.0.25"
#pipArgs+=" opencv-python==4.4.0.46"
# pipArgs+=" tensorflow-serving-api==1.15.0"
pipArgs+=" "


echo ___________________________________________________________________
echo conda install -c conda-forge -y -----${condaArgs}
conda install -c conda-forge -y ${condaArgs}

echo ___________________________________________________________________
echo pip install -------------------------${pipArgs}
pip install ${pipArgs}
