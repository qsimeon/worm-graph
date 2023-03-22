#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: _utils.py
@time: 2023/2/28 12:15
"""
from govfunc._pkg import *


def generate_polynomial(x, polyorder, usesine):
    # polyorder: polynomial formula with the first variant up to x^(i), where i is chosen from [1, polyorder]
    r, c = x.shape
    Theta = poolData(x, c, polyorder, usesine)
    return Theta


def neuro_plot(y, isTarget):
    y_df = pd.DataFrame(y)
    # data normalization: z-scoring
    cnt = 0
    interval = 10
    for i in range(0, y_df.shape[0]):
        y_df.iloc[i] = (y_df.iloc[i] - y_df.iloc[i].mean()) / y_df.iloc[i].std()
        y_df.iloc[i] = y_df.iloc[i] + cnt
        cnt -= interval
    # start plotting
    plt.figure(figsize=(6, 12))
    axe = plt.gca()
    axe.spines["top"].set_color("none")
    axe.spines["right"].set_color("none")
    axe.spines["left"].set_color("none")

    # transfer to list in order to re-label the y-axis
    list_y = []
    list_label = []

    for j in range(0, y_df.shape[0]):
        list_y.append(-j * interval)
        list_label.append(j)

    plt.ylabel("Neurons")
    plt.xlabel("Time(s)")

    plt.yticks(list_y, list_label, fontproperties="Times New Roman", size=6)

    for i in range(0, y_df.shape[0]):
        plt.plot(
            range(0, y_df.shape[1]),
            y_df.iloc[i],
            color=sns.color_palette("deep", n_colors=20)[i % 20],
            linewidth=0.5,
        )

    if isTarget == True:
        plt.savefig("./worm_response_target.png", dpi=1000, bbox_inches="tight")
    else:
        plt.savefig("./worm_response_pred.png", dpi=1000, bbox_inches="tight")
    plt.show()


def derivative(y, t):
    """
    input: [time, status]
    func: calculate the residual between time steps
    output: [residual(\delta t), status]
    """
    yrow, ycol = y.size()
    dy = np.zeros((yrow - 1, ycol))
    for i in range(0, yrow - 1):
        dy[i, :] = y[i + 1, :] - y[i, :]
    return dy


def poolData(yin, nVars, polyorder, usesine):
    """
    func: generate polynomial functions as candidates
    output: \theta(yin) as denoted in paper
    """
    n = yin.shape[0]
    yout = np.zeros((n, 1))

    ind = 0
    # poly order 0
    yout[:, ind] = np.ones(n)
    ind += 1

    # poly order 1
    for i in range(nVars):
        yout = np.c_[yout, np.ones(n)]
        yout[:, ind] = yin[:, i]
        ind += 1

    if polyorder >= 2:
        # poly order 2
        for i in range(nVars):
            for j in range(i, nVars):
                yout = np.c_[yout, np.ones(n)]
                yout[:, ind] = yin[:, i] * yin[:, j]
                ind += 1

    if polyorder >= 3:
        # poly order 3
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    yout = np.c_[yout, np.ones(n)]
                    yout[:, ind] = yin[:, i] * yin[:, j] * yin[:, k]
                    ind += 1

    if polyorder >= 4:
        # poly order 4
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    for l in range(k, nVars):
                        yout = np.c_[yout, np.ones(n)]
                        yout[:, ind] = yin[:, i] * yin[:, j] * yin[:, k] * yin[:, l]
                        ind += 1

    if polyorder >= 5:
        # poly order 5
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j, nVars):
                    for l in range(k, nVars):
                        for m in range(l, nVars):
                            yout = np.c_[yout, np.ones(n)]
                            yout[:, ind] = (
                                    yin[:, i]
                                    * yin[:, j]
                                    * yin[:, k]
                                    * yin[:, l]
                                    * yin[:, m]
                            )
                            ind += 1

    if usesine:
        for k in range(1, 11):
            yout = np.concatenate((yout, np.sin(k * yin), np.cos(k * yin)), axis=1)

    return yout


def sparsifyDynamics(Theta, dXdt, lam, n):
    """
    func: calculate coefficients of \theta() generated from poolData(...) using dynamic regression
    note: consume large computational resources
    """
    # Compute Sparse regression: sequential least squares
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]  # initial guess: Least-squares

    # lambda is our sparsification knob.
    for k in range(10):
        smallinds = np.abs(Xi) < lam  # find small coefficients
        Xi[smallinds] = 0  # and threshold
        for ind in range(n):  # n is state dimension
            biginds = ~smallinds[:, ind]
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(
                Theta[:, biginds], dXdt[:, ind], rcond=None
            )[0]

    return Xi


def governingFuncPredict(x0, Theta, Xi):
    x_hat = np.dot(Theta, Xi)
    # print(x_hat.shape, "---")
    # print(x_hat.shape)
    pred = calculas(x0, x_hat)
    return pred


def calculas(y0, y_hat):
    """
    this is the reverse of derivative
    """
    # print(y0.shape, y_hat.shape)
    yrow, ycol = y_hat.shape[0], y_hat.shape[1]
    sum_y = np.zeros((yrow + 1, ycol))
    sum_y[0, :] = y0
    for i in range(1, yrow + 1):
        sum_y[i, :] = sum_y[i - 1, :] + y_hat[i - 1, :]
    return sum_y


def coef_analysis(worm_name, n_cluster, folder):
    def plot_coefficient(data, sorted, w, path):
        plt.figure(figsize=(20, 20))
        sns.heatmap(data=data, square=True, cmap="RdBu_r", center=0, linecolor='grey', linewidths=0.3)
        if sorted:
            plt.title("(sourted) Coefficient of neurons activities of " + w)
            plt.savefig(os.path.join(path, 'sorted_coeff_' + w + '.png'))
        else:
            plt.title("(unsourted) Coefficient of neurons activities of " + w)
            plt.savefig(os.path.join(path, 'coeff_' + w + '.png'))

    def plot_dendrogram_scipy(clusters, labels):
        plt.figure(figsize=(30, 8))
        dendrogram = hierarchy.dendrogram(clusters, labels=labels, p=6, orientation="top", leaf_font_size=5,
                                          leaf_rotation=360)
        plt.ylabel('Euclidean Distance')
        plt.show()
        return dendrogram["ivl"], dendrogram["leaves_color_list"]

    # def plot_dendrogram_agg(model, w, **kwargs):
    #     # Create linkage matrix and then plot the dendrogram
    #
    #     # create the counts of samples under each node
    #     counts = np.zeros(model.children_.shape[0])
    #     n_samples = len(model.labels_)
    #     for i, merge in enumerate(model.children_):
    #         current_count = 0
    #         for child_idx in merge:
    #             if child_idx < n_samples:
    #                 current_count += 1  # leaf node
    #             else:
    #                 current_count += counts[child_idx - n_samples]
    #         counts[i] = current_count
    #
    #     linkage_matrix = np.column_stack(
    #         [model.children_, model.distances_, counts]
    #     ).astype(float)
    #     # Plot the corresponding dendrogram
    #     dendrogram(linkage_matrix, **kwargs)
    #     plt.title("dendrogram of " + w)
    #     plt.savefig('./govfunc/coefficient_CalToRes_tau_1/dendrogram_' + w + '.png')

    def main_function(w, n, folder):
        ###############################
        worm = w
        data = pd.read_hdf("./govfunc/Uzel2022/" + folder + "/coef_" + worm + ".hdf")
        path = os.path.dirname(os.path.abspath(__file__)) + "/Uzel2022/" + folder
        plot_coefficient(data, False, worm, path)

        data_np = data.to_numpy()
        index_col = data.columns.values.tolist()
        index_row = data.index.tolist()
        distance = data_np.T @ data_np
        # normalization
        distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))

        # # hierarchy.linkage
        # clusters = hierarchy.linkage(distance, method="complete")
        # label, leaf_catagory = plot_dendrogram_scipy(clusters, index_col)
        #
        # label_col = np.zeros_like(index_col)
        #
        #
        # for i in range(len(leaf_catagory)):
        #     item = int(leaf_catagory[i][-1])
        #     for j in range(len(index_col)):
        #         if index_col[j] == label[i]:
        #             label_col[j] = item
        # col_catagory = np.stack((index_col, label_col)).T
        # print(col_catagory)

        agg = AgglomerativeClustering(n_clusters=n, metric="precomputed", linkage="complete", compute_distances=True)
        agg.fit(distance)
        # plot_dendrogram_agg(agg, w, truncate_mode="level", p=n)
        label_col = np.array(agg.labels_, dtype=int)
        col_category = np.stack((index_col, label_col)).T

        # load the raw data
        graph_tensors = torch.load(os.path.join(ROOT_DIR, "data/processed/connectome", "graph_tensors.pt"))

        # make the graph
        graph = Data(**graph_tensors)
        dataset = load_Uzel2022()
        neuron_to_slot = dataset[worm]["neuron_to_slot"]
        real_catagory = []

        # mapping neuron index to its name
        for i in range(col_category.shape[0]):
            real_catagory.append(graph.y[neuron_to_slot.get(col_category[i, 0])].item())

        real_label = np.array(real_catagory, dtype=int).T.reshape(len(real_catagory), 1)
        not_sorted_result = np.concatenate((col_category, real_label), axis=1)
        # # result format [neuron_name, predict_category, real_category]
        # count = (not_sorted_result[:, 1] == not_sorted_result[:, 2]).sum()
        # print("count = ", count)

        # plt.scatter(range(0, not_sorted_result.shape[0]), not_sorted_result[:, 1])
        # plt.scatter(range(0, not_sorted_result.shape[0]), not_sorted_result[:, 2])
        # plt.legend(["predict", "target"], loc="lower right")
        # plt.show()

        # tagging the real label
        list_real_label = ["inter", "motor", "other", "pharynx", "sensory", "sexspec"]

        for i in range(not_sorted_result.shape[0]):
            not_sorted_result[i, 2] = list_real_label[int(not_sorted_result[i, 2])]

        # sorted by labels in prediction
        sorted_np = sorted(not_sorted_result, key=lambda x: x[1])
        data_list = list(data_np.T)
        _, sorted_data = zip(*sorted(zip(list(not_sorted_result[:, 1]), data_list), key=lambda x: x[0]))

        # list to numpy array and reshape
        sorted_np = np.array(sorted_np).reshape(len(sorted_np), 3)
        sorted_data = np.array(sorted_data).reshape(len(sorted_data), len(sorted_data[0])).T

        I = pd.Index(index_row, name="rows")
        C = pd.Index(sorted_np[:, 0], name="cols")
        data = pd.DataFrame(sorted_data, index=I, columns=C)
        plot_coefficient(data, True, w, path)
        return data, sorted_np

    data, sorted_np = main_function(worm_name, n_cluster, folder)
    return data, sorted_np
