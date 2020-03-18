# This sample uses a UFF MNIST model to create a TensorRT Inference Engine
# https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics
from random import randint
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    MODEL_FILE = "lenet5.uff"
    INPUT_NAME = "input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"


def trt_use():
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph_def,
        outputs=output_node_name,
        max_batch_size=batch_size,
        max_workspace_size_bytes=workspace_size,
        precision_mode=precision,
        minimum_segment_size=3)


def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = common.GiB(1)
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case(data_paths, pagelocked_buffer, case_num=randint(0, 9)):
    [test_case_path] = common.locate_files(data_paths, [str(case_num) + ".pgm"])
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)
    return case_num

def pb2uff():
    #     convert-to-uff frozen_inference_graph.pb
    import uff
    import tensorrt as trt
    print(uff.__path__)
    with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        parser.register_input("Placeholder", (1, 28, 28))
        parser.register_output("fc2/Relu")
    model_file = "frozen_inference_graph.uff"
    parser.parse(model_file, network)

def import_from_onnx():
    import tensorrt as trt
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_path, 'rb') as model:
            parser.parse(model.read())

    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1 << 20  # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.build_cuda_engine(
        network, config) as engine:


def from_Scratch():
    # Create the builder and network
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        # Configure the network layers based on the weights provided. In this case, the weights are imported from a pytorch model.
        # Add an input layer. The name is a string, dtype is a TensorRT dtype, and the shape can be provided as either a list or tuple.
        input_tensor = network.add_input(name=INPUT_NAME, dtype=trt.float32, shape=INPUT_SHAPE)

        # Add a convolution layer
        conv1_w = weights['conv1.weight'].numpy()
        conv1_b = weights['conv1.bias'].numpy()
        conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w,
                                        bias=conv1_b)
        conv1.stride = (1, 1)

        pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
        pool1.stride = (2, 2)
        conv2_w = weights['conv2.weight'].numpy()
        conv2_b = weights['conv2.bias'].numpy()
        conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
        conv2.stride = (1, 1)

        pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
        pool2.stride = (2, 2)

        fc1_w = weights['fc1.weight'].numpy()
        fc1_b = weights['fc1.bias'].numpy()
        fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

        relu1 = network.add_activation(fc1.get_output(0), trt.ActivationType.RELU)

        fc2_w = weights['fc2.weight'].numpy()
        fc2_b = weights['fc2.bias'].numpy()
        fc2 = network.add_fully_connected(relu1.get_output(0), OUTPUT_SIZE, fc2_w, fc2_b)

        fc2.get_output(0).name = OUTPUT_NAME
        network.mark_output(fc2.get_output(0))


def main():
    data_paths, _ = common.find_sample_data(description="Runs an MNIST network using a UFF model file",
                                            subfolder="mnist")
    model_path = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "models")
    model_file = os.path.join(model_path, ModelData.MODEL_FILE)

    with build_engine(model_file) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            case_num = load_normalized_test_case(data_paths, pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))


if __name__ == '__main__':
    main()
