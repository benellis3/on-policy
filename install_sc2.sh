#!/bin/bash
# Install SC2 and add the custom maps

pymarl_dir=$HOME/src/on-policy
cd $pymarl_dir/3rdparty/
smac_maps=$pymarl_dir/smac_maps

export SC2PATH="$pymarl_dir/3rdparty/StarCraftII"
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
        unzip -P iagreetotheeula SC2.4.10.zip
        rm -f SC2.4.10.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

cd ..
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip

cp -r "$smac_maps/*.SC2Map" ./SMAC_Maps 
mv SMAC_Maps $MAP_DIR
rm -f SMAC_Maps.zip


echo 'StarCraft II and SMAC are installed.'
