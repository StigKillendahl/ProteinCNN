# This file is part of the ProteinCNN project.
#
# @author Stig Killendahl & Kevin Jon Jensen
#
# Based on the OpenProtein framework, please see the LICENSE file in the root directory.

int_or_none = lambda x: int(x) if x is not None else None
str_to_bool = lambda x: x.lower() == "true"
int_list    = lambda x: [] if x == '' else [int(v.strip()) for v in x.split(",")]
float_list  = lambda x: [float(v.strip()) for v in x.split(",")]

class RunConfig():

    def __init__(self, file=None):
        config = {}
        if file is not None:
            config = self._create_map(file)
        self._create_config(config)

    def _create_map(self, filepath):
        configs = {}
        with open(filepath) as f:
            content = f.readlines()

        for line in content:

            if line[0] is "#" or line[0] == '\n':
                continue
            line = line.split("=")
            param, value = line[0], line[1]
            configs[param.strip()] = value.strip()
        return configs

    def _create_config(self, config):
        self.run_params = {
            "name":                 config.get("name", ""),
            "silent" :              str_to_bool(config.get("silent", True)),
            "hide_ui":              str_to_bool(config.get("hide_ui", False)),
            "evaluate_on_test":     str_to_bool(config.get("evaluate_on_test", False)),
            "min_updates":          int(config.get("min_updates", 1000)),
            "max_updates":          int(config.get("max_updates", 2000)),
            "minibatch_size":       int(config.get("minibatch_size", 1)),
            "eval_interval":        int(config.get("eval_interval", 10)),
            "learning_rate":        float(config.get("learning_rate", 0.01)),
            "max_sequence_length":  int(config.get("max_sequence_length", 2000)),
            "n_proteins":           int_or_none(config.get("n_proteins", None)),
            "use_evolutionary":     str_to_bool(config.get("use_evolutionary", True)),
            "training_file":        config.get("training_file","data/preprocessed/sample.txt"),
            "validation_file":      config.get("validation_file", "data/preprocessed/sample.txt"),
            "testing_file":         config.get("testing_file", "data/preprocessed/sample.txt"),
            "max_time":             int_or_none(config.get("max_time", None)),
            "c_epochs":             int(config.get("c_epochs", 0)),
            "only_angular_loss":    str_to_bool(config.get("only_angular_loss",False)),
            "angles_lr":            float(config.get("angles_lr", 0.01)),
            "loss_atoms":           config.get("loss_atoms", "c_alpha"),
            "weight_decay":         float(config.get("weight_decay", 0.0)),
            "betas":                float_list(config.get("betas","0.95,0.99" )),
            "print_model_summary":  str_to_bool(config.get("print_model_summary", "False")),
            "max_restarts":         int(config.get('max_restarts', 0)),
            "breakpoints":          int_list(config.get('breakpoints', ''))
        }

        self.model_params = {
            "architecture":         config.get("architecture", "cnn").lower(),
            "kernel":               int_list(config.get("kernel", "11")),
            "padding":              int_list(config.get("padding", "5")),
            "output_size":          int(config.get("output_size", 60)),
            "hidden_size":          int(config.get("hidden_size", 25)),
            "layers":               int(config.get("layers", 4)),
            "dilation":             int_list(config.get("dilation", "1")),
            "stride":               int_list(config.get("stride", "1")),
            "resnet_type":          config.get("resnet_type", "resnet34"),
            "channels":             int_list(config.get("channels","64,125,250")),
            "dropout":              float_list(config.get("dropout" ,"0.0,0.0"),),
            "spatial_dropout":      str_to_bool(config.get("spatial_dropout", "False"))
        }
