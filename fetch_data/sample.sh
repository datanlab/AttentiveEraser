mkdir -p ./DATA/sample/

# randomly sample images for test and vis
OUT=$(python3 /fetch_data/sampler.py)
echo ${OUT}

FILELIST=$(cat ./DATA/sample.txt)

for i in $FILELIST
do
    IMGID=${i#*/test-masks/}  
    IMGID=${IMGID%%_*}  
    LABELID=${i#*_}  
    LABELID=${LABELID%_*} 
    BOXID=${i##*_}  
    BOXID=${BOXID%.*}  

    $(cp ${i} ./DATA/sample/${IMGID}_${LABELID}_${BOXID}_mask.png)

    $(cp ./test/${IMGID}.jpg ./DATA/sample/${IMGID}_${LABELID}_${BOXID}.jpg)
done
