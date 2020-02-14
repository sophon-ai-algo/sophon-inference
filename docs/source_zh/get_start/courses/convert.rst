使用 bmnett 将 mobilenet 编译为 bmodel
______________________________________

我们已经在官网上传了一份 tensorflow mobilenet 的模型，直接下载即可：

    .. code-block:: shell

       wget https://sophon-file.bitmain.com.cn/sophon-prod/model/19/05/28/mobilenetv1_tf.tar.gz
       tar -zxvf mobilenetv1_tf.tar.gz


然后，使用下面的脚本将模型编译为 bmodel：

    .. code-block:: python

       #!/usr/bin/env python3
       import bmnett

       model_path = "mobilenetv1.pb"  # path of tensorflow frozen model, which to be converted.
       outdir = "bmodel/"             # path of the generated bmodel.
       target = "BM1684"              # targeted TPU platform, BM1684 or BM1682.
       input_names = ["input"]        # input operation names.
       output_names = ["MobilenetV1/Predictions/Reshape_1"]  # output operation names.
       shapes = [(1, 224, 224, 3)]    # input shapes.
       net_name = "mobilenetv1"       # name of the generated bmodel.

       bmnett.compile(model_path, outdir, target, input_names, output_names, shapes=shapes, net_name=net_name)


运行结束之后，在上面脚本指定的${outdir}目录下，会生成 compilation.bmodel.




