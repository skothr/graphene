#!/usr/bin/bash

IMAGE_DIR="./rendered"

PREFIX=$1    # e.g. PREFIX=test-sequence1 --> rendered/test-sequence1/test-sequence1-0000.hdr [...]
EXTENSION=$2
OUTPUT=$3


DEFAULT_EXTENSION=".png"
DEFAULT_OUTPUT="${PREFIX}.mp4"


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
RESOLUTION=`identify -format '%wx%h' ${F_PREFIX}-$(printf %05d 0)${EXTENSION}` # size of frame 0
echo " Frame 0 Size: ${RESOLUTION}"


# convert hdr files to png (TODO: ffmpeg HDR support?)
if [[ "${EXTENSION}" == *".hdr" ]]; then
    echo "Converting HDR images to PNG..."
    echo ""

    # get last file in sequence
    i=`printf %05d 0`
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

# lossy
#CMD="ffmpeg -i ${F_PREFIX}-%05d${EXTENSION}.png -c:v libx264 -vf fps=${FPS} -pix_fmt yuv420p -s 1920x1080 ${IMAGE_DIR}/${OUTPUT}"
# lossless
CMD="ffmpeg -i ${F_PREFIX}-%05d${EXTENSION} -c:v $CODEC -vf fps=$FPS -s ${RESOLUTION} -pix_fmt yuv420p10le -preset veryslow $IMAGE_DIR/$OUTPUT"

echo ""
echo "Making video..."
echo "  ==>  ${CMD}"
echo ""
echo ""

eval $CMD
