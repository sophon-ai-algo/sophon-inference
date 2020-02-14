## Example of SSD300 decoded by bm-ffmpeg, preprocessed by bm-ffmpeg, inference by sail.

## Usage:

* environment configuration on PCIE mode.

```shell
# set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${bmnnsdk2_path}/lib/ffmpeg/x86: \
       ${bmnnsdk2_path}/lib/decode/x86

# install sophon python whl
cd ${bmnnsdk2_path}/examples/sail/python3/x86
pip3 install sophon-x.x.x-py3-none-any.whl --user
```

* environment configuration on SOC mode.

```shell
# set PYTHONPATH
export $PYTHONPATH=${bmnnsdk2_path}/examples/sail/python3/soc/sophon:$PYTHONPATH
```

* A SSD example using bm-ffmpeg to decode and using bmcv to preprocess, with batch size is 1.

```shell
# bmodel: bmodel path, can be fp32 or int8 model
# input:  input path, can be image/video file or rtsp stream
# loops:  frames count to be detected, default: 1
python3 ./det_ssd_bmcv.py \
    --bmodel your-path-to-bmodel \
    --input your-path-to-input \
    --loops frames_count_to_detect
```

* A SSD example with batch size is 4 for acceleration of int8 model, using bm-ffmpeg to decode and using bmcv to preprocess.

```shell
# bmodel: bmodel path of int8 model
# input:  input path, video file or rtsp stream
# loops:  loop number of inference, each loop processes 4 frames. default: 1
python3 ./det_ssd_bmcv_4b.py \
    --bmodel your-path-to-bmodel \
    --input your-path-to-input \
    --loops frames_count_to_detect
```
