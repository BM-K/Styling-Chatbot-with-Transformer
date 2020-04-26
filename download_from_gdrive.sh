#!/bin/bash

#Usage: download_from_gdrive.sh FILEID FILENAME [isBig]
if [ "$3" == "" ]; then
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=$1' -O $2
else
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=$1" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$1" -o $2
fi



