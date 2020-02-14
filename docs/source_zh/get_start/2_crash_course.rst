小试牛刀：模型部署
=================

在本章中，我们将帮助你快速地将一个由 tensorflow 训练生成的 mobilenet 部署到 Sophon SC5 加速卡上。
在这之前，你需要准备一台安装了 Sophon SC5 加速卡的个人电脑，操作系统建议为 Ubuntu16.04。
另外，你还需要一份 BMNNSDK 的软件包。

本教程包含三个步骤。
首先，你需要安装驱动、bmnett、和sophon-inference。
这三个模块都在 BMNNSDK 软件包中。
然后，我们使用 bmnett 工具将 mobilenet 转换成 bmodel。
最后，我们使用 sophon-inference 把 bmodel 运行起来，
对一张图片进行分类。

.. toctree::
   :glob:

   courses/install
   courses/convert
   courses/deploy



