g++ -o gen_caffemodel gen_caffemodel.cpp -DCPU_ONLY=1 -I/work/code/cvitek_mlir/caffe/include/ -g -lglog -lprotobuf -lcaffe -lboost_system -L /work/code/cvitek_mlir/caffe/lib


#gen_caffemodel -m pd.prototxt -c pd.caffemodel
