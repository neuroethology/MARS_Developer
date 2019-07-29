# OPTIMIZED PIPELINE

Here are the file to export optimized model of detection and pose, and to test the new optimized pipeline

The models can be found [here](https://www.dropbox.com/home/team_folder/MARS/D_pipeline_opt_1.1)

Be sure to have installed tensorflow 1.1 by source as you will need as well the tensorflow folder to compile some files (in case).

**How to export/ freeze a trained model?**  
First we need to export or *freeze* the detection model. In this step we also want to reduce the number of file we need to make it work, so we will load the generated priors as well in the model end export those with it. 
All you need is `model_detection.py`, `priors_gen.pkl`, `model_detection.ckpt` and a script that those the job, in this case `export_detection.py`.
As mentioned the important files are the `.ckpt` ones. In the folder containing the trained model you have the file holding the 
weights `.data` and the `.meta` holding the graph and all its metadata. In production we just need the model and its weights
You will need to repeat this procedure for both mice model (black and white). Here I assume that the user will pass to the model an already 
preprocessed image. The export script will do the following:  
  * take as input an image already pre-processed (resized to 299 and normalized pixels values in [0,1])
  * converts all variables to contants
  * strip unused nodes in the graph
  * remove all traning specific nodes in the graph
  * remove batch normalization operations by folding them into the convolutions

To export the trained model, for instance for the black mouse, type the following:
```sh
python export_detection.py \
--checkpoint_path ./model_detection_black.ckpt-200000 \
--export_dit . \
--export_version 1
```
What this script does is:
* load the saved graph in the default graph and retrieve the graph_def
* restore the weights 
* remove all metadata useless for inference 
* save it to disk

This will create a protocol buffer file called `optimized_model_detection_black_1.pd`. This model will take as input
the images as a placeholder node. The output of the network are the location (where we already add the priors) the 
confidences of the predicted locations. The node names in this case are `predicted_locations` and `Multibox/Sigmoid`
as the confidences node.

Changing the model checkpoint and generated priors you can create in the same way the optimized model for the white mouse.

Then if you want to have a ligher model you can quantize the optimized model to a 8bit version.

**Why quantize?**  
Neural network models can take up a lot of space on disk, with the original AlexNet being over 200 MB in float format for
example. Almost all of that size is taken up with the weights for the neural connections, since there are often many 
millions of these in a single model. Because they're all slightly different floating point numbers, simple compression 
formats like zip don't compress them well. They are arranged in large layers though, and within each layer the weights 
tend to be normally distributed within a certain range, for example -3.0 to 6.0. The simplest motivation for quantization 
is to shrink file sizes by storing the min and max for each layer, and then compressing each float value to an eight-bit 
integer representing the closest real number in a linear set of 256 within the range. For example with the -3.0 to 6.0 range,
a 0 byte would represent -3.0, a 255 would stand for 6.0, and 128 would represent about 1.5. I'll go into the exact 
calculations later, since there's some subtleties, but this means you can get the benefit of a file on disk that's shrunk
by 75%, and then convert back to float after loading so that your existing floating-point code can work without any changes.
Another reason to quantize is to reduce the computational resources you need to do the inference calculations, by running
them entirely with eight-bit inputs and outputs. This is a lot more difficult since it requires changes everywhere you 
do calculations, but offers a lot of potential rewards. Fetching eight-bit values only requires 25% of the memory 
bandwidth of floats, so you'll make much better use of caches and avoid bottlenecking on RAM access. You can also 
typically use SIMD operations that do many more operations per clock cycle. In some case you'll have a DSP chip available 
that can accelerate eight-bit calculations too, which can offer a lot of advantages.
Moving calculations over to eight bit will help you run your models faster, and use less power (which is especially 
important on mobile devices). It also opens the door to a lot of embedded systems that can't run floating point code 
efficiently, so it can enable a lot of applications in the IoT world.

**How to quantize a model?**  
Make sure you did compile tensorflow from sourse by building the pip package from the tensorflow directory by 
```sh 
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt--mfpmath=both --copt=-msse4.1 --copt=-msse4.2 \
--config=cuda -k //tensorflow/tools/pip_package:build_pip_package
```

First you need to compile the tool. To do so, move to the tensorflow directory and run 
```sh 
bazel build tensorflow/tools/quantization:quantize_graph
```
Then you can actually quantize the optimized model by 
```sh 
bazel-bin/tensorflow/tools/quantization/quantize_graph \
--input=../mars/pipeline_opt_1.1/optimized_detection_black_1.pb \
--output_node_names="predicted_locations,Multibox/Sigmoid" \
--output=../mars/pipeline_opt_1.1/quant_detection_black.pd \
--mode=eightbit
```
This will produce a new model that runs the same operations as the original, but with eight bit calculations internally, 
and all weights quantized as well. The size now is about a quarted of the original.

However, for now, the quantized ops are supported just on CPU and indeed the inference will be much more slower compared to the 
optimized model. Further due to the quantization the results can be slightly different compared to the one you get from the 
optimized model. So for now I will use the optimized models and not the quantized ones.

**Export the pose trained model**
Now we have to do the same process of exporting the model for the pose model.

Here all you need is the checkpoint of the pose model and the `model_pose.py`. Here again we assume that the iamges rapresenting
the mouse in the bounding boxes resulting from the detection are already pre processed and ready to be passed as a placeholder node
to the Hourglass network.
As done before, we use an `export_pose.py` script to do this job where the output node will be the predicted_heatmaps by the
networks, from which we keep the last node names `HourGlass/Conv_30/BiasAdd`.
Then we can run the same bash command to export the model and if you want as well quantize it.

**How to use the frozen models and import multiple models in the same script**
In the case we have just one model, it would be very easy to just import the graph_def Protobuf first and load this graph_def into the actual Graph and run it on a session. Here, however we have basically to load three different models and to avoid conflicts between operations or nodes because we are importing everything into the default graph which is used in the session, so we have to keep them separate, meaning different graphs and different sessions.  
In order to do so, I created two classes, one for each model (detection, pose), used as utility to import the graphs. In this way this classes can be used to import as many graphs as you need. Each class has as well a `run` method that allow us to feed and run the inference.

The script that wrap all the graphs, pre processing, post processing a and basically run the entire pipeline is `optimized_pipeline.py`

## IN THIS FOLDER

|Filename | Description|
|---------|------------|
|export_detection.py | Utility to export the Multibox detection model|
|export_pose.py | utility to export the Hourglass pose model|
|optimized_pipeline.py| main script to run the pipeline|






