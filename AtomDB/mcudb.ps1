#! args field
Param
(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("validate", "analyze", "convert")]
    [string] $cmd,
    [Parameter(Mandatory = $true, Position = 1)]
    [ValidateSet("f412zg", "f413zh", "f746zg", "f767zi", "f769ni", "h743zi")]
    [string] $board,
    [Parameter(Mandatory = $true, Position = 2)]
    [string] $model
)

#! hardcode1 exec alias
Set-Alias stm32ai C:/Users/you/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/7.2.0/Utilities/windows/stm32ai.exe
Set-Alias stm32builder C:\ST\STM32CubeIDE_1.10.1\STM32CubeIDE\headless-build.bat
Set-Alias stm32prog 'C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe'

#! hardcode2 board serial number for customize flash and Comx
switch -Exact ($board) {
    'f412zg' { $SERIAL_NUMBER = 'serial_no'; $COMx = 'COM_no:115200' }
    'f746zg' { $SERIAL_NUMBER = 'serial_no'; $COMx = 'COM_no:115200' }
    'h743zi' { $SERIAL_NUMBER = 'serial_no'; $COMx = 'COM_no:115200' }
}
# Write-Output $SERIAL_NUMBER-$COMx

# global var defination
$DATE = Get-Date -Format "MMddHHmmss"
# todo add multi-network support
$MODEL_NAME = "network"
# todo add more compression type support
$MODEL_COMPRESSION = "none"
# set model type
if ( $model -like '*.tflite' ) {
    $MODEL_TYPE = "tflite"
}
elseif ( $model -like '*.onnx' ) {
    $MODEL_TYPE = "onnx"
}
#! hardcode3 path
$MCUPROFILER = "#this-repo-path#"
$WORKSPACE = "$MCUPROFILER\submodule\workspaces\$board\"
$WSL_PYTHON = "wsl-env-python3"
$WSL_IMAGENET = "IMGNET-trainset"

$PROJECT_DIR = "$MCUPROFILER\submodule\stm32projs\$board\"
$filename = Get-Item $PROJECT_DIR
$PROJECT_NAME = $filename.BaseName
$model_file_object = Get-Item $model
$model_file_name = $model_file_object.BaseName
$REPORT_DIR = "$MCUPROFILER\results\$board\$DATE-$model_file_name-$cmd"
$MODEL_FILE = "$REPORT_DIR/$MODEL_NAME.tflite"
$LOG_FILE = "$REPORT_DIR/profiler.log"

# wsl part
$WSL_MCUPROFILER = $MCUPROFILER -replace "\\", "/"
$WSL_MCUPROFILER = $WSL_MCUPROFILER -replace "D:", "/mnt/d"
$WSL_REPORT_DIR = $REPORT_DIR -replace "\\", "/"
$WSL_REPORT_DIR = $WSL_REPORT_DIR -replace "D:", "/mnt/d"
$WSL_MODEL_FILE = "$WSL_REPORT_DIR/$MODEL_NAME.tflite"
$WSL_LOG_FILE = "$WSL_REPORT_DIR/profiler.log"


# if cmd == convert, just convert
if ( $cmd -eq 'convert' ) {
    # set wsl onnx path
    $WSL_ONNX_MODEL = $model -replace "\\", "/"
    $WSL_ONNX_MODEL = $WSL_ONNX_MODEL -replace "D:", "/mnt/d"
    $REPORT_DIR = "$MCUPROFILER\results\convert"
    $MODEL_FILE = "$REPORT_DIR/$MODEL_NAME.tflite"
    $LOG_FILE = "$REPORT_DIR/profiler.log"
    # wsl part
    $WSL_MCUPROFILER = $MCUPROFILER -replace "\\", "/"
    $WSL_MCUPROFILER = $WSL_MCUPROFILER -replace "D:", "/mnt/d"
    $WSL_REPORT_DIR = $REPORT_DIR -replace "\\", "/"
    $WSL_REPORT_DIR = $WSL_REPORT_DIR -replace "D:", "/mnt/d"
    $WSL_MODEL_FILE = "$WSL_REPORT_DIR/$model.tflite"
    $WSL_MODEL_FILE = ""
    # convert onnx to tflite
    wsl $WSL_PYTHON $WSL_MCUPROFILER/utils/convert_tflite.py -d $WSL_IMAGENET -o $WSL_ONNX_MODEL -t $WSL_MODEL_FILE
    exit
}

# prepare folders
mkdir -p $REPORT_DIR

if ( $MODEL_TYPE -eq 'onnx' ) {
    # set wsl onnx path
    $WSL_ONNX_MODEL = $model -replace "\\", "/"
    $WSL_ONNX_MODEL = $WSL_ONNX_MODEL -replace "D:", "/mnt/d"
    # copy onnx model
    Copy-Item $model $REPORT_DIR
}
elseif ( $MODEL_TYPE -eq 'tflite' ) {
    # update model file
    $MODEL_FILE = $model
    # copy tflite model
    Copy-Item $model $REPORT_DIR
}


#! main workflow
if ( $MODEL_TYPE -eq 'onnx' ) {
    # convert onnx to tflite
    wsl $WSL_PYTHON $WSL_MCUPROFILER/utils/convert_tflite.py -d $WSL_IMAGENET -o $WSL_ONNX_MODEL -t $WSL_MODEL_FILE
}
if ( $? -and $cmd -eq 'validate' ) {
    # generate executable c files from tflite, always from tflite
    stm32ai generate -m $MODEL_FILE --name $MODEL_NAME --type tflite --compression $MODEL_COMPRESSION -o $PROJECT_DIR  -w $WORKSPACE --allocate-inputs --allocate-outputs
}
if ( $? -and $cmd -eq 'validate' ) {
    # build projs with new network
    stm32builder -data $WORKSPACE -cleanBuild $PROJECT_NAME
}
if ( $? -and $cmd -eq 'validate' ) {
    # flash elf to boards
    stm32prog -c port=swd sn=$SERIAL_NUMBER -d $PROJECT_DIR\Release\$PROJECT_NAME.elf -s
}
if ( $? -and $cmd -eq 'validate' ) {
    # validate model on boards
    stm32ai validate -m $MODEL_FILE --name $MODEL_NAME --workspace $REPORT_DIR\Memory --mode stm32 -d $COMx --allocate-inputs --allocate-outputs --output $REPORT_DIR -b 3
}
elseif ( $? -and $cmd -eq 'analyze' ) {
    # validate model on boards
    stm32ai analyze -m $model --name $MODEL_NAME --workspace $REPORT_DIR\Memory --allocate-inputs --allocate-outputs --output $REPORT_DIR
}
if ( $? ) {
    Write-Output "All done, check out $REPORT_DIR for more detail!"
}