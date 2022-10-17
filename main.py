import matplotlib.pyplot as plt
from data.data_loader import CElegansDataset
from tasks.all_tasks import OneStepPrediction
from models.gnn_models import EvolveRCGN
from train.train import optimize_model, model_predict

if __name__ == "__main__":
    graph = CElegansDataset()[0]
    task = OneStepPrediction(graph)
    model = EvolveRCGN(node_count=task.node_count, node_features=task.node_features)
    print('got here')
    model, log = optimize_model(task, model)
    # preds = model_predict(task, model)
    # plt.plot(preds[1, :task.train_size])
    # plt.show()