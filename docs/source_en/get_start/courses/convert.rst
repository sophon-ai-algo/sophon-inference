convert tensorflow frozen model to bmodel using bmnett
______________________________________________________
We have already uploaded the official tensorflow frozen model of mobilenetv1 on our website, just "wget" it!

    .. code-block:: shell

       wget https://sophon-file.bitmain.com.cn/sophon-prod/model/19/05/28/mobilenetv1_tf.tar.gz
       tar -zxvf mobilenetv1_tf.tar.gz

Then, convert tensorflow frozen model to bmodel using bmnett, as follow script:

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

After conversion, a bmodel named "compilation.bmodel" will generated under ${outdir}.




