# What is this for?

This folder should be used to store configuration files used by Kedro or by separate tools.

This file can be used to provide users with instructions for how to reproduce local configuration with their own credentials. You can edit the file however you like, but you may wish to retain the information below and add your own section in the [Instructions](#Instructions) section.

## Local configuration

The `local` folder should be used for configuration that is either user-specific (e.g. IDE configuration) or protected (e.g. security keys).

> *Note:* Please do not check in any local configuration to version control.

## Base configuration

The `base` folder is for shared configuration, such as non-sensitive and project-related configuration that may be shared across team members.

WARNING: Please do not put access credentials in the base configuration folder.

## Sample ./local/mlflow.yml config file:
```yml
# SERVER CONFIGURATION -------------------

# `mlflow_tracking_uri` is the path where the runs will be recorded.
# For more informations, see https://www.mlflow.org/docs/latest/tracking.html#where-runs-are-recorded
# kedro-mlflow accepts relative path from the project root.
# For instance, default `mlruns` will create a mlruns folder
# at the root of the project

# All credentials needed for mlflow must be stored in credentials .yml as a dict
# they will be exported as environment variable
# If you want to set some credentials,  e.g. AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# > in `credentials.yml`:
# your_mlflow_credentials:
#   AWS_ACCESS_KEY_ID: 132456
#   AWS_SECRET_ACCESS_KEY: 132456
# > in this file `mlflow.yml`:
# credentials: mlflow_credentials

server:
mlflow_tracking_uri: null # if null, will use mlflow.get_tracking_uri() as a default
mlflow_registry_uri: null # if null, mlflow_tracking_uri will be used as mlflow default
credentials: null  # must be a valid key in credentials.yml which refers to a dict of sensitive mlflow environment variables (password, tokens...). See top of the file.
request_header_provider: # this is only useful to deal with expiring token, see https://github.com/Galileo-Galilei/kedro-mlflow/issues/357
type: null # The path to a class : my_project.pipelines.module.MyClass. Should inherit from https://github.com/mlflow/mlflow/blob/master/mlflow/tracking/request_header/abstract_request_header_provider.py#L4
pass_context: False # should the class be instantiated with "kedro_context" argument?
init_kwargs: {} # any kwargs to pass to the class when it is instantiated

tracking:
# You can specify a list of pipeline names for which tracking will be disabled
# Running "kedro run --pipeline=<pipeline_name>" will not log parameters
# in a new mlflow run

disable_tracking:
pipelines: []

experiment:
name: SR5_experiment
restore_if_deleted: True  # if the experiment`name` was previously deleted experiment, should we restore it?

run:
id: null # if `id` is None, a new run will be created
name: second_run # if `name` is None, pipeline name will be used for the run name
nested: True  # if `nested` is False, you won't be able to launch sub-runs inside your nodes

params:
dict_params:
flatten: False  # if True, parameter which are dictionary will be splitted in multiple parameters when logged in mlflow, one for each key.
recursive: True  # Should the dictionary flattening be applied recursively (i.e for nested dictionaries)? Not use if `flatten_dict_params` is False.
sep: "." # In case of recursive flattening, what separator should be used between the keys? E.g. {hyperaparam1: {p1:1, p2:2}} will be logged as hyperaparam1.p1 and hyperaparam1.p2 in mlflow.
long_params_strategy: fail # One of ["fail", "tag", "truncate" ] If a parameter is above mlflow limit (currently 250), what should kedro-mlflow do? -> fail, set as a tag instead of a parameter, or truncate it to its 250 first letters?


# UI-RELATED PARAMETERS -----------------

ui:
port: "5000" # the port to use for the ui. Use mlflow default with 5000.
host: "127.0.0.1"  # the host to use for the ui. Use mlflow efault of "127.0.0.1".
```