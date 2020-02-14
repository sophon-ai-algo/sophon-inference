## Example of SSD300 decoded by bm-opencv, preprocessed by bm-opencv, inference by sail.

## Usage:

* Deploy on SOC mode.

```shell
make -f Makefile.arm

# bmodel: bmodel path, can be fp32 or int8 model
# input:  input path, can be image/video file or rtsp stream
# loops:  frames count to be detected, default: 1
./ssd300_cv_cv+bmcv_sail.arm \
    --bmodel your-path-to-bmodel \
    --input your-path-to-input \
    --loops frames_count_to_detect
```
