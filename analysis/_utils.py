#!/usr/bin/env python
# encoding: utf-8

from analysis._pkg import *
from visualization._utils import *

def find_config_files(root_dir):
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)
        if file == 'config.yaml':
            yield file_path
        elif os.path.isdir(file_path) and not file.startswith('.'):
            for config_file in find_config_files(file_path):
                yield config_file

def get_config_value(config, key):
    if '.' in key:
        key_parts = key.split('.')
        subconfig = config
        # import pdb; pdb.set_trace()
        for part in key_parts:
            if part in subconfig:
                subconfig = subconfig[part]
            else:
                return None
        return subconfig
    else:
        return config.get(key)


def plot_loss_vs_parameter(configs, param_names):
    time_stamp = 0
    for key in configs.keys():
        key_lists = key.split('/')
        time_stamp = str(key_lists[2])

    # Assuming you have loaded the config.yaml files into a dictionary named 'configs'
    param_values = {}
    for param_name in param_names:
        param_values[param_name] = set(get_config_value(config, param_name) for config in configs.values())

    # Compute trailing averages for each combination of parameter values
    trailing_averages = {}
    for param_values_combination in itertools.product(*param_values.values()):
        matching_configs = {}
        for config_path, config in configs.items():
            if all(get_config_value(config, param_name) == param_value for param_name, param_value in zip(param_names, param_values_combination)):
                matching_configs[config_path] = config

        if not matching_configs:
            continue

        trailing_avg_values = []
        for config_path, _ in matching_configs.items():
            loss_path = os.path.join(config_path, 'loss_curves.csv')
            if os.path.exists(loss_path):
                loss_df = pd.read_csv(loss_path)
                num_worms = len(os.listdir(config_path)) - 3 # Subtract 3 for .hydra, config.yaml, and loss_curves.csv
                trailing_avg = loss_df['centered_test_losses'][-num_worms:].mean()
                trailing_avg_values.append(trailing_avg)

        param_values_tuple = tuple(param_values_combination)
        trailing_averages[param_values_tuple] = sum(trailing_avg_values) / len(trailing_avg_values)

    # Plot trailing averages as scatter plot with regression line and confidence intervals
    plt.figure()
    plt.xlabel(', '.join(param_names))
    plt.ylabel('Mean Trailing Avg. Validation Loss')
    plt.title('Mean Trailing Avg. Validation Loss vs. {}'.format(', '.join(param_names)))
    param_values_tuples = list(trailing_averages.keys())
    param_values_lists = [list(t) for t in param_values_tuples]
    sns.regplot(x=param_values_lists, y=list(trailing_averages.values()), scatter=True, ci=95, order=2, label='Data')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd()+"/analysis/figures/", "val_loss_on_[" + str(time_stamp) + "].png"))
    plt.show()


def plot_loss_vs_parameter_sorted(configs, param_names):
    '''
    Plots the mean trailing average validation loss for each combination of parameter values.
    Args:
        configs: range of experiments
        param_names: showing on x-axis
    Returns:
        figure val_loss vs param_names

    '''
    time_stamp = 0
    for key in configs.keys():
        key_lists = key.split('/')
        time_stamp = str(key_lists[2])

    # Assuming you have loaded the config.yaml files into a dictionary named 'configs'
    param_values = {}
    for param_name in param_names:
        param_values[param_name] = set(get_config_value(config, param_name) for config in configs.values())

    # Compute trailing averages for each combination of parameter values
    trailing_averages = {}
    for param_values_combination in itertools.product(*param_values.values()):
        matching_configs = {}
        for config_path, config in configs.items():
            if all(get_config_value(config, param_name) == param_value for param_name, param_value in zip(param_names, param_values_combination)):
                matching_configs[config_path] = config

        if not matching_configs:
            continue

        trailing_avg_values = []
        for config_path, _ in matching_configs.items():
            loss_path = os.path.join(config_path, 'loss_curves.csv')
            if os.path.exists(loss_path):
                loss_df = pd.read_csv(loss_path)
                # calculate the number of worm
                num_worms = len([item for item in os.listdir(config_path) if item.startswith('worm')])
                trailing_avg = loss_df['centered_test_losses'][-num_worms:].mean()
                trailing_avg_values.append(trailing_avg)

        param_values_tuple = tuple(param_values_combination)
        trailing_averages[param_values_tuple] = sum(trailing_avg_values) / len(trailing_avg_values)

    # Plot trailing averages as scatter plot with regression line and confidence intervals
    plt.figure()
    plt.xlabel(', '.join(param_names))
    plt.ylabel('Mean Trailing Avg. Validation Loss')
    plt.title('Mean Trailing Avg. Validation Loss vs. {}'.format(', '.join(param_names)))
    param_values_tuples = list(trailing_averages.keys())
    param_values_lists = [list(t) for t in param_values_tuples]
    sorted_x, sorted_y = zip(*sorted(zip(param_values_lists, list(trailing_averages.values()))))
    plt.plot(sorted_x, sorted_y, label='loss')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd()+"/analysis/figures/", "val_loss_on_[" + str(param_names[0]) + "_" + str(time_stamp) + "].png"))
    plt.show()