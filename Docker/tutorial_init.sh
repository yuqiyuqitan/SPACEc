echo
echo "Starting SpacCodEx environment"
echo "=============================="

# TODO: tryin to fix issue with extracted files keeing ids from inside the archive
#echo "Setup for user: ${RUN_UID}"
#useradd --uid ${RUN_UID} spacodex
#mkdir /home/spacodex
#chown -R spacodex:spacodex /home/spacodex
#chown -R ${RUN_ID}:${RUN_ID} /workspace

echo "Checking tutorial materials."
cd /workspace

# if data and notebook folder do not exist
if [ ! -d "/workspace/data" ] && [ ! -d "/workspace/notebooks" ]; then
	echo "Data and notebooks for the tutorial are missing."
	read -p "Do you want to download the tutorial materials? [y/n] " -n 1 -r
	echo
	if [[ ! $REPLY =~ ^[Yy]$ ]]; then
		echo "Skipping download."
	else
		echo
		echo "Downloading tutorial materials:"
		wget -O materials.zip "https://www.dropbox.com/scl/fo/qxgrxq6y7y2jlrwhotabd/h?rlkey=3rp5cjkkrexnfxkqbtgr7l3h6&dl=1"
		unzip materials.zip
		rm materials.zip
		echo
		echo "Downloading model:"
		cd notebooks
		mkdir models
		cd models
		mkdir Mesmer_model
		cd Mesmer_model
		wget -O model.tar.gz "https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-9.tar.gz"
		tar -xzvf model.tar.gz
		rm model.tar.gz
		chown -R ${UID}:${UID} .
		cd ../../..
	fi
else
	echo "Data and notebooks folders for the tutorial are present. Skipping download."
fi

echo
echo "Starting Jupyter Lab:"
${MAMBA_EXEC} run -n ${ENV_NAME} jupyter lab --allow-root --NotebookApp.token="" --ip="*"
