<br/>**Q:** What Sophon products are the samples running on?</br>
<br/>**A:** SC3 for now. More details please refer to https://sophon.ai</br>

<br/>**Q:** Where to download BMSDK?</br>
<br/>**A:** https://sophon.cn/drive/40.html</br>

<br/>**Q:** What are the samples?</br>
<br/>**A:** They are deep learning applications for deployment. All samples are in domain of Computor Version for now.Such as Image Classification, Object Detection, Face Detection and Semantic Segmentation.</br>

<br/>**Q:** What is bmodel?</br>
<br/>**A:** Models are trained in framworks such as tensorflow. Before deploying on Sophon products, the models should be compiled by compilation tools such as bmentc/bmnett/bmnetd/bmnetm/bmnetp. The complied result is the so called IR(intermediate representation). We named it bmodel.</br>

<br/>**Q:** "BMSDK not found" when excuting 'camke ..'.</br>
<br/>**A:** Make BMSDK was installed. It contains 'bmlang bmnetc bmnetm bmnetp bmnett driver include lib and so on. Then set the absolute path of the installed BMSDK diretory in CMakeLists.txt.</br>

<br/>**Q:** "bmnetc: command not found" when converting bmodels.</br>
<br/>**A:** Go to the install directory of BMSDK and 'source envsetup.sh'.</br>
