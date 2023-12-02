echo "Starting SpacCodEx environment"
echo "=============================="

# if data and notebook folder do not exist
if [ ! -d "/workspace/data" ] && [ ! -d "/workspace/notebooks" ];; then
    echo "Data and notebooks for the tutorial are missing."
    read -p "Do you want to download the tutorial materials? [y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        echo "Skipping download."
    else
        echo "Downloading tutorial materials"
        wget -O materials.zip "https://www.dropbox.com/scl/fo/qxgrxq6y7y2jlrwhotabd/h?rlkey=3rp5cjkkrexnfxkqbtgr7l3h6&dl=1"
        unzip materials.zip
        rm materials.zip
    fi
else
    echo "Data and notebooks folders for the tutorial are present. Skipping download."
fi

${MAMBA_EXEC} run -n ${ENV_NAME} jupyter lab --allow-root --NotebookApp.token="" --ip="*"