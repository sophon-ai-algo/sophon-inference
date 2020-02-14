Usage
_____


Get model and data
^^^^^^^^^^^^^^^^^^

    To run this demo, we need a fp32 bmodel of mtcnn.
    We also need a face image to be detected.
    We can get them through the script "download.py". 

        .. code-block:: shell
          
           python3 download.py mtcnn_fp32.bmodel 
           python3 download.py face.jpg

Run C++ cases
^^^^^^^^^^^^^

    For case 0:
    
        .. code-block:: shell

           ./det_mtcnn --bmodel ./mtcnn_fp32.bmodel --input ./face.jpg


Run python cases
^^^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

           python3 ./det_mtcnn.py --bmodel ./mtcnn_fp32.bmodel --input ./face.jpg



