# CONFIG FILES
Configuration files are used for specifying model and hyper parameters for the run, each parameter follows the format:

```
# Comment, will not be parsed
parameter_name = <value>
``` 

Values can be of different types. Below are a list of parameters, their type and a short description. For all parameters default value is also specified.

## RUN PARAMS

| Parameter | type | Description | Default Value |
| --- | --- | --- | --- |
| name | string | run name used for experimentID| "" |
| hide_ui | boolean | If true, ProteinCNN will not start webserver for live data plotting | False |
| min_update | int | minimum number of minibatches processed before training can terminate  | 1000 |
| max_update | int | training terminate if max_update  == minibatches processed | 2000 |
| minibatch_size | int | ... | 1 |
| eval_interval | int | number of training steps before model is evaluated again on validation dataset | 10 |
| learning_rate | float | model learning rate at each training step | 0.01 |
| max_sequence_length | int | max protein sequence length used in for run | 2000 |
| use_evolutionary | boolean | if true, the input will consist of the amino-acids and their PSSM values | True |
| training_file | string | raw data file used for training samples | '' |
| validation_file | string | raw data file used for evaluation model | '' |
| max_time | int | maximum time (in seconds). If sat the training will terminate if the time used reaches it | None |
| angles_lr | float | specify unique learning rate for the learned angle matrix | 0.01 |
| loss_atoms | string | either 'all' or 'c_alpha'. Determines if drmsd loss is calculated on the c_alpha backbone molecyle or all molecyles in the backbone. | 'c_alpha' |
| weight_decay |float | specify the weight decay of the optimizer (AdamOptimizer) | 0.0 |
| betas | float,float | specify betas for the optimizer | [0.95,0.99] |
| print_model_summary | boolean | prints are complete model summary of the network (does not work for RNN) | False |
| milestones | [(int,int),...] | specify validation milestones for the run. Each entry is given as a tuple '(100,12.5),(500,11.3)' | None |
| restart_on_failed_milestone | boolean | restarts model training if milestone is not reached, same parameters but using new seed | False |
| max_restarts | int | maximum number of run restarts | 0 |

## MODEL PARAMS
| Parameter | type | Description | Default Value |
| --- | --- | --- | --- |
| architecture | string | specify which architecture is used; either [cnn,rnn,resnet] | cnn |
| dropout | [int,int] | specifies input dropout and dropout between layers | [0.5,0.5] |
| output_size | int | output_size for the model, a alphabet of same size will be created for angle classification | 60 |


## CNN PARAMS
| Parameter | type | Description | Default Value |
| --- | --- | --- | --- |
| kernel | [int,...?] | kernel for each convolution layer, if single value is given it will be used for all layers | 11 |
| spatial_dropout | boolean | specify if spatial_dropout is used instead of normal dropout | False |
| layers | int | number of layers | 4 |
| dilation | [int,...?] | dilation for each convolution layer, if single value is given it will be used for all layers | 1 |
| stride | [int,...?]| stride for each convolution layer, if single value is given it will be used for all layers | 1 |
| channels | [int,...?] | channels for each convolution layer, if single value is given it will be used for all layers | 64 |
| padding | [int,...?] | padding for each convolution layer, if single value is given it will be used for all layers. padding should be the (kernel-1)/2 on each layer | 5 |



## BIDIRECTIONAL LSTM PARAMS
| Parameter | type | Description | Default Value |
| --- | --- | --- | --- |
| hidden_size | int | number of hidden neurons for each LSTM layer | 200 |
| | | | |

## PRESNET PARAMS
| Parameter | type | Description | Default Value |
| --- | --- | --- | --- |
| resnet_type | string | specify type of Presnet from ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] | resnet34 |
| kernel | [int,int] | kernel used for basisblocks and bottlenecks of PresNet | 11 |
| padding | [int,...?] | padding for basicblocks and bottlenecks. Padding should be the (kernel-1)/2 | 5 |



## EXAMPLE CONFIG FILE
```
#Example run config
silent=True
hide_ui=True
name=cnn_700_51k
evaluate_on_test=False
min_updates=20000
max_updates=20000
minibatch_size=32
eval_interval=100
learning_rate=0.0001
angle_lr=0.001
max_sequence_length=700
training_file=training_95
validation_file=validation
use_evolutionary=True
max_time=259200
loss_atoms=c_alpha

#Model parameters example for CNN
architecture=CNN
layers=5
output_size=60
#Specify kernel and padding for each layer or 1 value for same for all layers
kernel=51
padding=25
channels=200,500,500,200
dropout=0.5,0.5
#Dilation and stride can also be specified but will default to 1 for all layers


```
