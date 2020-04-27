## Summary

```shell
se5_tests
├── cls_resnet                  # test cases of resnet50 for classification(build from cpp)
│   ├── cls_resnet_0
│   └── cls_resnet_1
├── det_ssd                     # test case of ssd for detection(python)
│   ├── det_ssd_1.py
│   └── det_ssd_2.py
├── README.md
└── scripts                     # contains test scripts
     ├── auto_test.sh
     └── download.py
```

## Notes

```shell
# the only thing you should do.
# the script will download needed models and data in directory: ./data
# add then test all the cases, it will exit when bug occurs.
./scripts/auto_test.sh
```
