#!/usr/bin/bash

IMAGE_DIR="./rendered"

DEFAULT_EXTENSION=".png"
DEFAULT_OUTPUT="output.mp4"


PREFIX=$1    # e.g. PREFIX=test-sequence1 --> rendered/test-sequence1/test-sequence1-0000.hdr [...]
EXTENSION=$2
OUTPUT=$3

# check PREFIX
if [ "${PREFIX}" == "" ]; then
    echo "Error: No prefix defined!"
    exit 1
fi
# check EXTENSION
if [ "${EXTENSION}" == "" ]; then
    echo "No extension defined (using default)"
    EXTENSION=".hdr"
elif [[ ! "${EXTENSION}" == *"." ]]; then
    EXTENSION=".${EXTENSION}"
fi
# check OUTPUT
if [ "${OUTPUT}" == "" ]; then
    echo "No output file defined (using default)"
    OUTPUT=${DEFAULT_OUTPUT}
fi

# print parameters
echo ""
echo " ==> Prefix:    ${PREFIX}"
echo " ==> Extension: ${EXTENSION}"
echo " ==> Output:    ${OUTPUT}"
echo " ==> Directory: ${IMAGE_DIR}/${PREFIX}/"
echo ""
echo " ====> ${IMAGE_DIR}/${PREFIX}/${PREFIX}-%05d.png"
echo ""

F_PREFIX="${IMAGE_DIR}/${PREFIX}/${PREFIX}"

# convert hdr files to png (TODO: ffmpeg HDR support?)
if [[ "${EXTENSION}" == *".hdr" ]]; then
    echo "Converting HDR images to PNG..."
    echo ""

    # get last file in sequence
    i=0000
    while true; do
        if test -f "${F_PREFIX}-${i}${EXTENSION}"; then
            i=$((10#$i+1))
            i=`printf "%05d" $i`
        elif [ $i == 0 ]; then
            echo "Error: no image files with provided prefix! ( ${F_PREFIX}-${i}${EXTENSION} )"
            exit 1
        else  
            i=$((10#$i-1))
            i=`printf "%05d" $i`
            LAST_FILE=`ls ${F_PREFIX}-${i}${EXTENSION}`
            echo "Last file: in sequence: ${LAST_FILE}"
            break
        fi
    done

    NUM=${LAST_FILE#"${PREFIX}-"} # remove prefix from filename
    echo "$NUM"
    NUM=${NUM//".hdr"/}  # ${EXTENSION#"."}}    # remove extension from filename
    echo "$NUM"
    NUM=$((10#$NUM))
    
    START=0
    END=`printf "%05d" ${NUM}`
    for i in `seq $START $END`; do
        i=`printf "%05d" $i`

        if test -f "${F_PREFIX}-${i}.png"; then
            echo "  ==>  (found ${F_PREFIX}-${i}.png)"
        else
            CMD="convert ${F_PREFIX}-${i}${EXTENSION} ${F_PREFIX}-${i}.png"
            echo "  ==>  $CMD"
            convert ${F_PREFIX}-${i}${EXTENSION} ${F_PREFIX}-${i}.png
        fi
    done
    EXTENSION=".png"
fi


CODEC="libx265"

#CMD="ffmpeg -i ${F_PREFIX}-%05d${EXTENSION}.png -c:v libx264 -vf fps=${FPS} -pix_fmt yuv420p ${IMAGE_DIR}/${OUTPUT}"

# lossless test
CMD="ffmpeg -i ${F_PREFIX}-%05d${EXTENSION} -c:v $CODEC -vf fps=$FPS -s 1920x1080 -pix_fmt yuv420p10le -preset veryslow $IMAGE_DIR/$OUTPUT"

echo ""
echo "Making video..."
echo "  ==>  ${CMD}"
echo ""
echo ""

eval $CMD

#ffmpeg -i ${IMAGE_DIR}/${F_PREFIX}-%05d${EXTENSION} -c:v libx264 -vf fps=${FPS} -pix_fmt yuv420p ${IMAGE_DIR}/${OUTPUT}

# (old command with more options):
#    ffmpeg -i stable-explosion-3-%04d.png -c:v libx264 -s:v 2048x2048 -vf fps=24 -pix_fmt yuv420p stable-explosion-3.mp4
