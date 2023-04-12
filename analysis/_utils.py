#!/usr/bin/env python
# encoding: utf-8

from analysis._pkg import *
from visualization._utils import *
from train._utils import *
import utils


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
            if all(get_config_value(config, param_name) == param_value for param_name, param_value in
                   zip(param_names, param_values_combination)):
                matching_configs[config_path] = config

        if not matching_configs:
            continue

        trailing_avg_values = []
        for config_path, _ in matching_configs.items():
            loss_path = os.path.join(config_path, 'loss_curves.csv')
            if os.path.exists(loss_path):
                loss_df = pd.read_csv(loss_path)
                num_worms = len(os.listdir(config_path)) - 3  # Subtract 3 for .hydra, config.yaml, and loss_curves.csv
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
    plt.savefig(os.path.join(os.getcwd() + "/analysis/figures/", "val_loss_on_[" + str(time_stamp) + "].png"))
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
            if all(get_config_value(config, param_name) == param_value for param_name, param_value in
                   zip(param_names, param_values_combination)):
                matching_configs[config_path] = config

        if not matching_configs:
            continue

        trailing_avg_values = []
        for config_path, _ in matching_configs.items():
            loss_path = os.path.join(config_path, 'loss_curves.csv')
            if os.path.exists(loss_path):
                loss_df = pd.read_csv(loss_path)
                trailing_avg = loss_df['centered_test_losses'].min()
                trailing_avg_values.append(trailing_avg)

        param_values_tuple = tuple(param_values_combination)
        trailing_averages[param_values_tuple] = sum(trailing_avg_values) / len(trailing_avg_values)

    # Plot trailing averages as scatter plot with regression line and confidence intervals
    plt.figure()
    plt.xlabel(', '.join(param_names))
    plt.ylabel('Trailing Validation Loss')
    plt.title('Trailing Validation Loss vs. {}'.format(', '.join(param_names)))
    param_values_tuples = list(trailing_averages.keys())
    param_values_lists = [list(t) for t in param_values_tuples]
    sorted_x, sorted_y = zip(*sorted(zip(param_values_lists, list(trailing_averages.values()))))
    plt.plot(sorted_x, sorted_y, label='loss')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd() + "/analysis/figures/",
                             "val_loss_on_[" + str(param_names[0]) + "_" + str(time_stamp) + "].png"))
    plt.show()


def neurons_on_category(model, dataset, smooth_data, tau_out):
    '''
    predict neuron activities using pre-trained model

    Args:
        model: pre-trained model
        dataset: whole dataset (multi worms)
        smooth_data: Boolean, choose whether to use smoothed data or not
        tau_out: offset to the future prediction
    Returns:
        Tuple, (worm_name, {category1: {max, min, mean}, category2: {max, min, mean}, .....})
        the result losses(max, min, average) of all worms with categorized neurons
    Figures:
        save the figures about the loss based on neuron category, across worms
    '''

    neuron_type = ["inter", "motor", "other", "pharynx", "sensory", "sexspec"]

    model.eval()

    # load the category of neurons: len(graph_tensors) == 302
    graph_tensors = torch.load(os.path.join(ROOT_DIR, "data/processed/connectome", "graph_tensors.pt"))
    graph = Data(**graph_tensors)

    # TODO: add criterion as one of the hyperparameters in this function
    criterion = torch.nn.L1Loss()

    multi_worm_category_loss = dict()
    for i in range(len(dataset)):
        single_worm_dataset = dataset["worm" + str(i)]
        if smooth_data:
            calcium_data = single_worm_dataset["smooth_calcium_data"]
        else:
            calcium_data = single_worm_dataset["calcium_data"]
        max_time = single_worm_dataset["max_time"]
        named_neuron_mask = single_worm_dataset["named_neurons_mask"]

        named_neuron_inds = torch.where(named_neuron_mask)[0].numpy()

        # get the type of named neurons
        category_neuron = [graph.y[item].item() for item in named_neuron_inds]

        # make predictions with final model
        targets, predictions = model_predict(model, calcium_data[:max_time // 2, :] * named_neuron_mask, tau=tau_out)
        loss_prediction = [criterion(predictions[:, idx], targets[:, idx]).detach().item() for idx in named_neuron_inds]
        # print(len(named_neuron_inds), len(loss_prediction))

        # named_neuron_category_loss is a dict
        # key: (neuron_inds, loss)
        # val: neuron_category
        named_neuron_category_loss = dict()
        for k in range(len(named_neuron_inds)):
            named_neuron_category_loss.setdefault(
                (named_neuron_inds[k], loss_prediction[k]),
                category_neuron[k]
            )

        # reverse_dict is a dict
        # key: neuron_category
        # val: (neuron_inds, loss)
        reverse_dict = defaultdict(list)
        for key, val in sorted(named_neuron_category_loss.items()):
            reverse_dict[val].append(key)

        one_worm_category_neuron_loss = dict()
        for res_catagory in reverse_dict.items():
            loss_list = [loss for (neuron, loss) in res_catagory[1]]
            max_loss = np.array(loss_list).max()
            min_loss = np.array(loss_list).min()
            avg_loss = np.array(loss_list).mean()
            one_worm_category_neuron_loss.setdefault(
                res_catagory[0],
                {
                    "avg_loss": avg_loss,
                    "max_loss": max_loss,
                    "min_loss": min_loss,
                }
            )

        multi_worm_category_loss["worm" + str(i)] = one_worm_category_neuron_loss

    # plot the average & max & min loss of each worm with categorized neurons
    for worm in multi_worm_category_loss.items():
        sorted_category = sorted(list(worm[1]))
        min_loss_list = [worm[1][idx]["min_loss"] for idx in sorted_category]
        max_loss_list = [worm[1][idx]["max_loss"] for idx in sorted_category]
        avg_loss_list = [worm[1][idx]["avg_loss"] for idx in sorted_category]

        sns.lineplot(
            x=sorted_category,
            y=avg_loss_list,
            label="avg. " + worm[0],
            alpha=0.5,
        )

        plt.gca().fill_between(
            sorted_category,
            min_loss_list,
            max_loss_list,
            alpha=0.1,
            # label=worm[0],
        )
    plt.xticks(range(6), neuron_type)
    plt.legend(loc="upper right", fontsize=2)
    plt.ylabel("categorized prediction loss")
    plt.xlabel("category of neurons")
    plt.title("Analysis on neuron-type and prediction-loss \n Dataset: " + dataset["worm0"]["dataset"])
    plt.savefig(os.path.join(ROOT_DIR, "analysis", "figures",
                             "category_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png"))

    return multi_worm_category_loss


def plot_trailing_loss_vs_parameter_legend(config_pardir, parameter, legend):
    '''
    Plot the trailing loss vs parameter with legend
    Args:
        config_pardir: the parent directory of the config files that you want to plot
        parameter: specify the parameter on the x-axis, e.g. "model_params.hidden_size"
        legend: specify the parameter on legend, e.g. "dataset.name"

    Returns:
        None

    '''

    configs = {}
    # go through all the config files and get the parameters and the loss
    for file_path in find_config_files(config_pardir):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            parent_dir = os.path.dirname(file_path)
            if os.path.exists(os.path.join(parent_dir, "loss_curves.csv")):
                loss_df = pd.read_csv(os.path.join(parent_dir, "loss_curves.csv"), index_col=0)
                loss = loss_df["centered_test_losses"][loss_df["centered_test_losses"].idxmin()]
                configs[os.path.dirname(file_path)] = (loss, OmegaConf.create(data))

    parameters = parameter.split(".")
    param_name = parameters[1]
    legends = legend.split(".")

    # get the trailing loss for each parameter value
    trailing_dict = {}
    for key, value in configs.items():
        legend_name = value[1][legends[0]][legends[1]]
        param_value = value[1][parameters[0]][parameters[1]]
        loss = value[0]
        print(f"{legend_name} {param_value} {loss}")
        if legend_name in trailing_dict.keys():
            trailing_dict[legend_name].update({param_value: loss})
        else:
            trailing_dict[legend_name] = {param_value: loss}

    # plot the trailing loss for each parameter value
    x = []
    for para_n, val_dict in trailing_dict.items():
        sorted_x, sorted_y = zip(*sorted(zip(list(val_dict.keys()), list(val_dict.values()))))
        x = sorted_x
        plt.plot(sorted_x, sorted_y, label=para_n)

    plt.legend()
    plt.xlabel(str(param_name) + str(x))
    plt.ylabel("Trailing Loss")
    plt.title("Analysis on " + str(param_name) + " \n Trailing Losses for Different " + str(legends[1]) + "s")
    plt.savefig("analysis/figures/trailing_loss_on_" + str(parameter) + "_legend_" + str(legend) + ".png")
