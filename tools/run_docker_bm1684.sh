docker run -it --rm \
       -v `pwd`/sophon-inference:/workspace/sophon-inference \
       -v `pwd`/release:/workspace/bm168x \
       -v `pwd`/soc_bm1684_asic:/workspace/soc_bm1684_asic \
       -v `pwd`/nntoolchain-bm1684-all-9.9.9:/workspace/nntoolchain \
       bmnnsdk2-bm1684/dev:2.0.0 \
       bash
