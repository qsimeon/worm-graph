from data.load_connectome import CElegansDataset
from data.load_neural_activity import load_Uzel2022
from tasks.all_tasks import OneStepPrediction
from models.gnn_models import EvolveRCGN
from train.train import optimize_model
from train.train import model_predict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # laod conectome
    graph = CElegansDataset()[0]
    # load data
    Uzel2022 = load_Uzel2022(); single_worm_dataset = Uzel2022['worm1']
    dataset, neuron_ids = single_worm_dataset['data'], single_worm_dataset['neuron_ids']
    # define the task
    task = OneStepPrediction(graph, dataset, neuron_ids)
    # construct the model
    model = EvolveRCGN(node_count=graph.num_nodes, node_features=task.seq_len)
    # train the model
    model, log = optimize_model(task, model)
    preds = model_predict(task, model)
    # TODO: put plotting code like this in a separate module file
    plt.figure()
    plt.plot(log['epochs'], log['train_losses'], label='train')
    plt.plot(log['epochs'], log['test_losses'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('EvolveRCGN loss curves')
    plt.legend()
    plt.show()
    # # plot prediction for a random neuron
    # plt.figure()
    # nid = np.random.choice(task.node_count)
    # sc = preprocessing.MinMaxScaler()
    # data = sc.fit_transform(task.graph.x.clone().detach().numpy())
    # plt.plot(data[nid, :1000], alpha=0.3, label='real')
    # plt.gca().set_prop_cycle(None)
    # plt.plot(preds[nid, :1000], '--', label='predicted', linewidth=3) 
    # plt.xlabel('Time')
    # plt.ylabel('Normalized Voltage')
    # plt.title('EvolveRCGN prediction for neuron %s'%nid)
    # plt.legend()
    # plt.show()