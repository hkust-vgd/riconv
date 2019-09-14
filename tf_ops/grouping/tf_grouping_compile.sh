# #/bin/bash
# PYTHON=python3
# CUDA_PATH=/usr/local/cuda
# TF_LIB=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
# PYTHON_VERSION=$($PYTHON -c 'import sys; print("%d.%d"%(sys.version_info[0], sys.version_info[1]))')
# TF_PATH=$TF_LIB/include
# $CUDA_PATH/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -L$TF_LIB -ltensorflow_framework -I $TF_PATH/external/nsync/public/ -I $TF_PATH -I $CUDA_PATH/include -lcudart -L $CUDA_PATH/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0



#/bin/bash
PYTHON=python3
CUDA_ROOT=/usr/local/cuda
TF_ROOT=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

${CUDA_ROOT}/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4 or above
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0