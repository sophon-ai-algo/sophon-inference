# **Algorithm Toolkit**

## 1. Algorithm Toolkit Info
> Toolkit Basic / Algorithm Info

### 1.1. Algokit Basic Info
  * pysail version: 1.1.3
  * supported chip mode: BM1682
  * supported data precision: FP32

### 1.2. Algokit Algorithm Info

  | Models | Caffe | TensorFlow | MXNet | PyTorch |
  |:------:|:-----:|:----------:|:-----:|:-------:|
  | mobilenetssd | √ | - | - | - |
  | yolov3 | √ | - | √ | - |
  | mobilenetyolov3 | √ | - | - | - |
  | fasterrcnn | √ | √ | - | - |
  | mtcnn | √ | - | - | - |
  | ssh | √ | - | - | - |
  | googlenet | √ | - | - | - |
  | resnet50 | √ | - | - | √ |
  | resnext50 | - | - | √ | - |
  | vgg16 | √ | - | - | - |
  | mobilenetv1 | √ | √ | - | - |
  | deeplabv3 | - | √ | - | - |


## 2. Algokit Test
> Supported Python Version: 2/3

### 2.1. Test Algorithm Module

```bash
cd ${sophon-inference}/tests/algokit_unittests
# test face detection mtcnn
python3 test_mtcnn.py
# test face detection ssh
python3 test_ssh.py
# test object detection yolov3
python3 test_yolov3.py
# test object detection yolov3
python3 test_yolov3_mx.py
# test object detection mobilenetyolov3
python3 test_mobilenetyolov3.py
# test object detection ssd
python3 test_mobilenetssd.py
# test object detection fasterrcnn
python3 test_fasterrcnn.py
# test object detection fasterrcnn_resnet50_tf
python3 test_fasterrcnn_resnet50_tf.py
# test general classification
python3 test_generalclassifier.py
# test semantic segmentation
python3 test_deeplabv3_mobilenetv2_tf.py
```

### 2.2. Test Algorithm Module According To Input Parameters

```bash
cd ${sophon-inference}/samples/python
# view algorithm list
python3 run_algokit.py --check_list
# choose a algorithm type && algorithm name
python3 run_algokit.py \
    --algo_type ['cls', 'obj_det', 'face_det', 'seg'] \
    --algo_name ['googlenet', 'vgg16', 'resnet50', 'mobilenetv1',
                 'mobilenetv1_tf', 'resnext50_mx', 'resnet50_pt',
                 'yolov3', 'mobilenetssd', 'mobilenetyolov3',
                 'fasterrcnn_vgg', 'fasterrcnn_resnet50_tf',
                 'yolov3_mx', 'mtcnn', 'ssh', 'deeplabv3_mobilenetv2_tf'] \
    --input_path 'input_file_path' \
    --vis # open visualization (default closed)
```

## 3. Algokit Usage
> How to use it && How to add an algorithm module

### 3.1. Run an algorithm module

* Creates the specified algorithm object

 ```python
 # Import the specified type algorithm constructor
 # Import object detection algorithm constructor
 from algokit.algofactory.objectdetection_factory import ObjectDetector
 # Import face detection algorithm constructor
 from algokit.algofactory.facedetection_factory import FaceDetector
 # Import general classification algorithm constructor
 from algokit.algofactory.generalclassification_factory import GeneralClassifier
 # Import object detection algorithm which will need to be split with autodeploy
 from algokit.algofactory.autodeploy_factory import ObjectDetector
 # Import semantic segmentation algorithm which will need to be split with autodeploy
 from algokit.algofactory.autodeploy_factory import SemanticSegment

 # Create the specified algorithm object (e.g. GeneralClassifier)
 classifier = GeneralClassifier().create(ClassificationModel.RESNET50, ChipMode.BM1682)
 ```

 * Run the algorithm module
 ```python
 # Test the specified algorithm (e.g. classifier)
 out = classifier.predict(input)
 ```

 ### 3.2. Add an algorithm module (e.g. mobilenetssd)

* Creates a new class for the specified algorithm

```python
# MobilenetSSD belong to CV detection algorithm
# Create the new class in algokit/algo_cv/det
# The parent class BaseEngine contains some basic data processing operations
class ObjectDetectionMOBILENETSSD(BaseEngine):
    """Construct mobilenetssd detector
    """
    def __init__(self, param1, param2, ...):
        super(ObjectDetectionMOBILENETSSD, self).__init__()
        self.xxx = xxx # member variable init
        self.net = Engine(model_description_dict, chip_mode) # Construct TPU inference engine
    def mobilenetssd_postprocess(self, ...):
        """
        processing after inference e.g. detection output layer
        """
        pass
    def mobilenetssd_preporcess(self, ...):
        """
        different frameworks and types of algorithm preprocess are different
        or call the base operation of the parent class
        """
        pass
    def xxx(self, ...):
        """
        additional operation
        """
        pass
```

* Add corresponding branch in the algorithm factory

```python
# Add the new branch in algokit/algofactory/objectdetection_factory
# 1. Add branch in create method
if detection_model is ObjDetModel.MOBILENETSSD:
    from ..algo_cv.det.object_detection_mobilenetssd import \
        import ObjectDetecitonMOBILENETSSD as Detector

# 2. Add global DEFAULT_XXX_PARAM
DEFAULT_MOBILENETSSD_PARAM = {
    'detected_size': (300, 300),
    'threshold': 0.25,
    'nms_threshold': 0.45,
    'num_classes': 21,
    'priorbox_num': 1917
}
# 3. Add param loader branch in load_param method
if default_param is ObjDetModel.MOBILENETSSD:
    default_param = DEFAULT_MOBILENETSSD_PARAM
```

* Add a new algorithm type in algokit global config

```python
# Add the new algorithm type in algokit/kitconfig
class ObjDetModel(Enum):
  """The object detection algorithm model
  """
  ...
  ...
  MOBILENETSSD = 'mobilenetssd'
```

* Add a new model description config file

```shell
# Add the new model description config file in algokit/engine/engineconfig
# Create new json file: bm1682_mobilenetssd.json
{
  "arch": {
    "context_path": "mobilenetssd_ir/compilation.bmodel", # bmodel path suffix，base path: ${HOME}/.sophon/models
    "is_dynamic": false, # the inputs size of model is variable
    "tpus": "0", # tpu id
    "input_names": ["data"], # the input names of model
    "output_names": ["mbox_conf_flatten", "mbox_loc", "mbox_priorbox"], # the output names of model
    "input_shapes": [[1, 3, 300, 300]] # the default input size of model
  }
}
```