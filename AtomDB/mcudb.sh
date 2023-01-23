#!/bin/bash
# This shell script update mcudb.
# Author:
#     Anonymous
# History:
#     2022.09.17 First Release
# Dependencies:
#     None
# Attention:
#     1. Read comments.

function Usage() {
    echo "mcudb"
    echo ""
    echo "mcudb.sh [-h] [-v] [-b <board_name>] [-c <validate|analyze>] [-m <model_file>] "
    echo ""
    echo "-h: show this guide."
    echo "-v: show this tool version."
    echo ""
    echo "example: "
    echo "    mcudb.sh -b h743zi -c validate -m ./test.tflite"
}

function main() {
    # Default setting
    model=''
    cmd='validate'
    board='h743zi'
    number=0

    while getopts :hvb:c:m:n: varname; do
        case $varname in
        h)
            Usage
            exit
            ;;
        v)
            VERSION='0.0.1'
            echo "Version==${VERSION}"
            exit
            ;;
        b)
            board=${OPTARG}
            ;;
        c)
            cmd=${OPTARG}
            ;;
        m)
            model=${OPTARG}
            ;;
        n)
            number=${OPTARG}
            ;;
        :) # option is None
            # set default
            case ${OPTARG} in
            b)
                echo "No specific board, exit now."
                exit 1
                ;;
            c)
                echo "You must set cmd, exit now."
                exit 1
                ;;
            m)
                echo "You must set model, exit now."
                exit 1
                ;;
            n)
                echo "You must set number, exit now."
                exit 1
                ;;
            esac
            ;;
        ?) # invalid option
            echo "Invalid option: -$OPTARG"
            Usage
            exit 2
            ;;
        esac
    done

    if [ -z $model ]; then
        exit 1
    fi

    #! hardcode1 alias, serial number, serial comunicate file or port
    stm32ai='/prefix/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/7.2.0/Utilities/linux/stm32ai'
    stm32builder='/opt/st/stm32cubeide_1.10.1/headless-build.sh'
    stm32prog='/prefix/usr/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI'
    python='/prefix/usr/miniconda3/envs/mcu/bin/python3'

    case $board in
    'f412zg')
        case $number in
        0)
            SERIAL_NUMBER='0676FF495252717267224355'
            COMx='/dev/f412zg'
            ;;
        1)
            SERIAL_NUMBER='066DFF343334434257145729'
            COMx='/dev/f412zg_1'
            ;;
        esac
        ;;
    'f746zg')
        case $number in
        0)
            SERIAL_NUMBER='0672FF525750877267011746'
            COMx='/dev/f746zg'
            ;;
        1)
            SERIAL_NUMBER='066EFF525254667867170844'
            COMx='/dev/f746zg_1'
            ;;
        esac
        ;;
    'h743zi')
        case $number in
        0)
            SERIAL_NUMBER='066DFF494849887767091431'
            COMx='/dev/h743zi'
            ;;
        1)
            SERIAL_NUMBER='066CFF484851897767064215'
            COMx='/dev/h743zi_1'
            ;;
        esac
        ;;
    esac

    # global var definition
    DATE=$(date +%m%d%H%M%S)
    # todo add multi-network support
    MODEL_NAME="network"
    # todo add more compression type support
    MODEL_COMPRESSION="none"
    model_file_name=$(basename ${model})
    MODEL_TYPE=${model##*.}
    #! hardcode2 path part
    MCUPROFILER="#this-repo-AtomDB-path#"
    IMAGENET="IMGNET-trainset"

    if [ ! $number -eq 0 ]; then
        WORKSPACE="$MCUPROFILER/submodule/workspaces/${board}_${number}/"
        PROJECT_DIR="$MCUPROFILER/submodule/stm32projs/${board}_${number}/"
        PROJECT_NAME=${board}_${number}
    else
        WORKSPACE="$MCUPROFILER/submodule/workspaces/$board/"
        PROJECT_DIR="$MCUPROFILER/submodule/stm32projs/$board/"
        PROJECT_NAME=$board
    fi

    REPORT_DIR="$MCUPROFILER/results/$board/$DATE-$model_file_name-$cmd"
    MODEL_FILE="$REPORT_DIR/$MODEL_NAME.tflite"

    # prepare folders
    mkdir -p $REPORT_DIR
    echo 'report_dir is '${REPORT_DIR}
    cp $model $REPORT_DIR

    #! main workflow
    if [ "$MODEL_TYPE" = "onnx" ]; then
        # convert onnx to tflite
        $python $MCUPROFILER/utils/convert_for_latency.py -d $IMAGENET -o $model -t $MODEL_FILE
    else
        cp $model $MODEL_FILE
    fi
    if [ $? -eq 0 -a "$cmd" = "validate" ]; then
        # generate executable c files from tflite, always from tflite
        $stm32ai generate -m $MODEL_FILE --name $MODEL_NAME --type tflite --compression $MODEL_COMPRESSION -o $PROJECT_DIR -w $WORKSPACE --allocate-inputs --allocate-outputs
    elif [ ! "$cmd" = "analyze" ]; then
        exit 2
    fi
    if [ $? -eq 0 -a "$cmd" = "validate" ]; then
        # build projs with new network
        $stm32builder -data $WORKSPACE -cleanBuild $PROJECT_NAME
    elif [ ! "$cmd" = "analyze" ]; then
        exit 3
    fi
    if [ $? -eq 0 -a "$cmd" = "validate" ]; then
        # flash elf to boards
        $stm32prog -c port=swd sn=$SERIAL_NUMBER -d $PROJECT_DIR/Release/$PROJECT_NAME.elf -s
    elif [ ! "$cmd" = "analyze" ]; then
        # add OOFM-flag, s.t. OutOfFlash, OutOfMem
        mv $model ${model}-OOFM
        exit 4
    fi
    if [ $? -eq 0 -a "$cmd" = "validate" ]; then
        # validate model on boards
        $stm32ai validate -m $MODEL_FILE --name $MODEL_NAME --workspace $REPORT_DIR/Memory --mode stm32 -d $COMx --allocate-inputs --allocate-outputs --output $REPORT_DIR -b 3
    elif [ "$cmd" = "analyze" ]; then
        $stm32ai analyze -m $MODEL_FILE --name $MODEL_NAME --workspace $REPORT_DIR/Memory --allocate-inputs --allocate-outputs --output $REPORT_DIR --verbosity 1
    fi
}

main "$@"
