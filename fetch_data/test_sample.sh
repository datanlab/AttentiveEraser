mkdir -p /hy-tmp/DATA/sample3/

# randomly sample images for test and vis
OUT=$(python3 /hy-tmp/MyREMOVAL/fetch_data/sampler.py)
echo ${OUT}

FILELIST=$(cat /hy-tmp/DATA/sample3.txt)

for i in $FILELIST
do
    IMGID=${i#*/test-masks/}  # 删除前缀
    IMGID=${IMGID%%_*}  # 删除第一个_及其后面的内容
    LABELID=${i#*_}  # 删除前缀
    LABELID=${LABELID%_*}  # 删除最后一个_及其后面的内容
    BOXID=${i##*_}  # 删除最后一个_及其前面的内容
    BOXID=${BOXID%.*}  # 删除后缀

    # 复制原始文件并重命名为 imgid_labelid_boxid_mask.png
    $(cp ${i} /hy-tmp/DATA/sample3/${IMGID}_${LABELID}_${BOXID}_mask.png)

    # 复制 /hy-tmp/test/imgid.jpg 文件
    $(cp /hy-tmp/test/${IMGID}.jpg /hy-tmp/DATA/sample3/${IMGID}_${LABELID}_${BOXID}.jpg)
done
