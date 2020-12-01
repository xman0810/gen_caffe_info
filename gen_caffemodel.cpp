#include <iostream>
#include <cstring>
#include <map>
#include <string>
#include <random>
#include <vector>
#include <fstream>
#include <getopt.h>
#include <caffe/caffe.hpp>

using namespace caffe;

static char *prototxt = NULL;
static char *caffemodel = NULL;
static bool gen_Prototxt = false;
static bool gen_Caffemodel = false;
static bool forward = false;

class DumpBlob {
public:
  DumpBlob(const BlobProto& blob) : blob_(blob) {}
  friend std::ostream& operator<<(std::ostream& os, const DumpBlob& d) {
    os << "[ ";
    for (auto dim : d.blob_.shape().dim())
      os << dim << " ";
    os << "]";
    return os;
  }
private:
  const BlobProto& blob_;
};

static int parse_arg(int argc, char** argv) {
  int opt;

  const char *optstring = "p:c:PCF";
  struct option long_options[] = {
      {"prototxt", required_argument, 0, 'p'},
      {"caffemodel", required_argument, 0, 'c'},
      {"gen_Prototxt", no_argument, 0, 'P'},
      {"gen_Caffemodel", no_argument, 0, 'C'},
      {"forward", no_argument, 0, 'F'},
      {0,0,0,0}
  };

  int option_index = 0;

  while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {
    switch(opt) {
    case 'p':
      printf("prototxt %s\n", optarg);
      prototxt = optarg;
      break;
    case 'c':
      printf("caffemodel %s\n", optarg);
      caffemodel = optarg;
      break;
    case 'P':
      printf("gen Prototxt true\n");
      gen_Prototxt = true;
      break;
    case 'C':
      printf("gen Caffemodel true\n");
      gen_Caffemodel = true;
      break;
    case 'F':
      printf("Forward true\n");
      forward = true;
      break;

    default:
      return -1;
    }
  }

  return 0;
}

void usage(void) {
    std::cout << "Invalide arguments" << std::endl;
    std::cout << "  gen_caffemodel.bin " << std::endl
              << "  --prototxt=prototxt" << std::endl
              << "  --caffemodel=caffemodel" << std::endl
              << "  --gen_Caffemodel" << std::endl
              << "  --gen_Prototxt" << std::endl;
}

static void fill_input(caffe::Net<float> *net) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0, 1);

  Blob<float>* input_blob = net->input_blobs()[0];

  int count = 1;
  count *= input_blob->channels();
  count *= input_blob->width();
  count *= input_blob->height();

  net->Reshape();

  auto data = input_blob->mutable_cpu_data();

  for (int k = 0; k < count; k++) {
    data[k] = distribution(generator);
  }
}

static int blob_count(const BlobProto& blob) {
  int count = 1;
  for (auto dim : blob.shape().dim()) {
    count *= dim;
  }
  return count;
}

int generate_caffemodel(const string deploy_prototxt, const string dst_caffemodel) {
  NetParameter net_param;
  ReadProtoFromTextFile(deploy_prototxt, &net_param);

  caffe::Net<float> *net = new caffe::Net<float>(deploy_prototxt, TEST);

  for (int i = 0; i < net_param.layer_size(); i++) {
    auto *param = net_param.mutable_layer(i);
    Layer<float> *layer = net->layer_by_name(param->name()).get();

    std::cout << "name:" << param->name() << ", type:" << param->type() << std::endl;

    if (layer->blobs().size() > 0) {
      param->clear_blobs();
      for (auto src : layer->blobs()) {
        BlobProto *blob = param->add_blobs();
        for (auto dim : src->shape()) {
          blob->mutable_shape()->add_dim(dim);
        }
      }
    }

    if (param->type() == "BN") {
      auto* scale = param->mutable_blobs(0);
      for (int k = 0; k < blob_count(*scale); k++) {
        scale->add_data(0.5);
      }

      auto* shift = param->mutable_blobs(1);
      for (int k = 0; k < blob_count(*shift); k++) {
        shift->add_data(0.101);
      }

      auto* mean = param->mutable_blobs(2);
      for (int k = 0; k < blob_count(*mean); k++) {
        mean->add_data(1);
      }

      auto* var = param->mutable_blobs(3);
      for (int k = 0; k < blob_count(*var); k++) {
        var->add_data(0.3);
      }
    } else if (param->type() == "BatchNorm") {
      std::cout << param->name() << ", " << param->type() << " blobs:\n";

      BlobProto *blob_mean = param->mutable_blobs(0);
      for (int k = 0; k < blob_count(*blob_mean); k++) {
        blob_mean->add_data(1);
      }
      std::cout << "\t" << DumpBlob(param->blobs(0)) << std::endl;

      BlobProto *blob_var = param->mutable_blobs(1);
      for (int k = 0; k < blob_count(*blob_var); k++) {
        blob_var->add_data(0.001);
      }
      std::cout << "\t" << DumpBlob(param->blobs(1)) << std::endl;

      BlobProto *blob_fraction = param->mutable_blobs(2);
      for (int k = 0; k < blob_count(*blob_fraction); k++) {
        blob_fraction->add_data(1);
      }
      std::cout << "\t" << DumpBlob(param->blobs(2)) << std::endl;

    } else if (param->blobs_size() > 0) {
      std::cout << param->name() << ", " << param->type() << " blobs:\n";

      for (int i = 0; i < param->blobs_size(); i++) {
        auto *blob = param->mutable_blobs(i);

        std::cout << "\t" << DumpBlob(*blob) << std::endl;

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0, 0.05);

        for (int k = 0; k < blob_count(*blob); k++) {
          float number = distribution(generator);
          if (number < -1.0)
            number = -1.0;
          if (number > 1.0)
            number = 1.0;
          blob->add_data(number);
        }
      }
    }
  }

  WriteProtoToBinaryFile(net_param, dst_caffemodel);
  return 0;
}

void gen_prototxt() {
  NetParameter proto;
  ReadProtoFromBinaryFile(caffemodel, &proto);
  WriteProtoToTextFile(proto, prototxt);
}

void do_forward() {
  std::shared_ptr<Net<float> > net_;
  net_.reset(new Net<float>(prototxt, TEST));
  net_->CopyTrainedLayersFrom(caffemodel);
  net_->Forward();
  const boost::shared_ptr<Blob<float>> blob = net_->blob_by_name("p1_concat_mbox_conf_perm");
  auto data = (blob.get())->cpu_data();

  std::cout << "p1_concat_mbox_conf_perm: elem 0: " << data[0]<< std::endl;
  std::cout << "p1_concat_mbox_conf_perm: elem 1: " << data[1]<< std::endl;

  const boost::shared_ptr<Blob<float>> blob_conv = net_->blob_by_name("conv_blob64");
  auto data_conv = (blob_conv.get())->cpu_data();
  if (data_conv) {
    std::cout << "conv64: elem 0: " << data_conv[0]<< std::endl;
    std::cout << "conv64: elem 1: " << data_conv[1]<< std::endl;
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging("");
  google::SetStderrLogging(google::GLOG_INFO);

  int ret = parse_arg(argc, argv);
  if (ret || !prototxt || !caffemodel) {
    usage();
    exit(-1);
  }

  std::cout << "<CMD> ";
  for (int i = 0; i < argc; i++) {
    std::cout << argv[i] << " ";
  }
  std::cout << std::endl;

  if (gen_Caffemodel) {
    generate_caffemodel(prototxt, caffemodel);
  }
  if (gen_Prototxt) {
    gen_prototxt();
  }
  if (forward) {
    do_forward();
  }


  return 0;
}
