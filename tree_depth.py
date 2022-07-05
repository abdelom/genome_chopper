import numpy as np
from detect_breakpoints import internal_incompatibility_2, detect_events
import matplotlib.pyplot as plt
from msprime_simulation import msprime_simulate_variants, test_tsinfer
from sklearn.linear_model import LinearRegression
import tsdate as td


class Site:

    def __init__(self, position):
        self.position = position


class Variant_m:

    def __init__(self, position):
        self.site = Site(position)


def rsquared(data, dict_reg, dict_pred=None):
    for key in data:
        reg = LinearRegression(fit_intercept=False).fit(data[key][0].reshape(len(data[key][0]), 1),
                                                        data[key][1].reshape(len(data[key][0]), 1))
        dict_reg[key].append(reg.score(data[key][0].reshape(len(data[key][0]), 1), data[key][1].reshape(len(data[key][0]), 1)))
        if dict_pred:
            dict_pred[key] = reg.predict(data[key][0].reshape(len(data[key][0]), 1))
    return dict_reg, dict_pred


def msprime_depth(tree_sequence):
    """

    :param tree_sequence: objet tree sequence de la suite tskit
    :return: liste des temps de coalescence, date des neouds du ts
    """
    return [node.time for node in tree_sequence.tables_dict["nodes"] if node.time > 0.0]


########################################################################################################################
##########################################estimateur##################################################################
########################################################################################################################


def depth_tmrca(variants, partition, sample_size, k):
    """

    :param variants: list d'objets Variants de la suite tskit, liste des sites polymorphes
    :param partition: sous échantillon des individues de l'aligenement
    :param sample_size: le nombre d'individues total
    :param k: la taille du sous échantillon, pour le stem, ou la taille du sous échantillon - 1, pour l crown
    :return: depth, float, la profondeur, au taux de mutation près, de du tmrca au sous échantillon
    """
    depth = 0
    length = variants[-1].site.position - variants[0].site.position
    individuals = np.array(range(sample_size))
    partition = set(partition)
    for variant in variants[1:-1]:
        partition_tmp = set(individuals[variant.genotypes == 1])
        if partition_tmp.union(partition) == partition:
            depth += len(partition_tmp)
    return depth / (length * k)


def crown_stem(crown, inf, sup, variants, partition, partition_pos):
    if crown:
        delete = np.array(partition_pos)
        delete = delete[delete > inf]
        delete = delete[delete < sup]
        delete = delete - inf
        variants_tmp = np.delete(variants[inf: sup + 1], delete)
        return depth_tmrca(variants_tmp, partition, len(variants[0].genotypes), len(partition))
    else:
        return depth_tmrca(variants[inf: sup], partition, len(variants[0].genotypes), len(partition))


########################################################################################################################
##########################################hierarchical bloc#############################################################
########################################################################################################################


########################################################################################################################
##########################################maximal bloc##################################################################
########################################################################################################################


def partition_position(variants, sample_size):
    """

    :param variants: list d'objets Variants de la suite tskit, liste des sites polymorphes
    :param sample_size: le nombre d'individues total
    :return:partition_dict: dictionnaire, les clé sont les partitions engendrés par les snps et les valeurs du dictinnaire
    les listes des snp générant les partition
    """
    individuals = np.array(range(sample_size))
    partition_dict = {}
    for index, variant in enumerate(variants):
        partition = tuple(individuals[variant.genotypes == 1])
        if len(partition) < 2:
            continue
        if partition in partition_dict:
            partition_dict[partition].append(index)
        else:
            partition_dict[partition] = [index]
    return partition_dict


def internal_incompatibilities(partition_dict, partition):
    """

    :param partition_dict: dictionnaire, les clé sont les partitions engendrés par les snps et les valeurs du dictinnaire
    :param partition: une partition corespondantz à un snp rencontré au moins une fois dans l'alignement
    :return: la liste des positions de tout les snp incompatibles avec la partition courante
    """
    list_incompatibilities = []
    for partition_2 in partition_dict:
        if internal_incompatibility_2(set(partition), set(partition_2)): # si les
            list_incompatibilities += partition_dict[partition_2]
    return sorted(list_incompatibilities)


def clade_in_block(list_incompatibilities, partition_pos):
    """

    :param list_incompatibilities:liste des sites incompatible avec la ti courante.
    :param partition_pos: liste des position de la partition courante
    :return:list de tuples délimitant des bloques dans lequel la partition est réalisée
    """
    list_blocks = []
    inf = list_incompatibilities[0]
    i = 0
    for sup in list_incompatibilities[1:]:
        while i < len(partition_pos) and partition_pos[i] < inf:
            i += 1
        if i == len(partition_pos):
            break
        if inf < partition_pos[i] < sup:
            list_blocks.append((inf, sup))
        inf = sup
    return list_blocks


def depths_clade(partition, partition_dict, variants, crown):
    """

    :param partition: partition courante
    :param partition_dict: dictionnaire, les clé sont les partitions engendrés par les snps et les valeurs du dictinnaire
    :param variants:  liste d'objet Variants du module msprime
    :param crown: boolean, si true calcul du crown sinon du stems
    :return: list des tmrca pour chaque bloque pour sous echantillon définie par la partition dans klequel la partition correspondante est réalisé
    """
    list_depth = []
    list_born = []
    list_incompatibilities = [elem for elem in internal_incompatibilities(partition_dict, partition) if elem]
    for inf, sup in clade_in_block(list_incompatibilities, partition_dict[partition]):
        if inf is not None:
            list_depth.append(crown_stem(crown, inf, sup, variants, partition, partition_dict[partition]))
            list_born.append((inf, sup))
    list_depth = np.array(list_depth)
    return np.array(list_born)[list_depth > 0], list_depth[list_depth > 0]


def depths_clades_size(variants, crown, sample_size, size):
    partition_dict = partition_position(variants, sample_size)
    list_depth_size = [np.array([])] * sample_size
    for partition in partition_dict:
        tmp = np.array(depths_clade(partition, partition_dict, variants, crown)[1])
        list_depth_size[len(partition)] = np.concatenate((list_depth_size[len(partition)], tmp))
    if size:
        list_stat = []
        for list_depth in list_depth_size:
            if not list(list_depth):
                continue
            list_stat.append(summary_stat(list_depth))
        return list_stat
    return np.concatenate(list_depth_size)


def summary_stat(list_h):
    """
    distributions des longueurs de blocs
    """
    length_distribution = [0] * 21
    lc = np.mean(list_h)
    nb_blocks = len(list_h)
    i = 0
    for length in list_h:
        index = (length / lc) // 0.2
        if index < 20:
            length_distribution[int(index)] += 1
        else:
            length_distribution[-1] += 1
        i += 1
    return np.array(length_distribution) / nb_blocks, lc


########################################################################################################################
##########################################msprime vs estimate###########################################################
########################################################################################################################


def clade_time_msprime(tree, root):
    if tree.is_leaf(root):
        return [([root], {"time": 0})]
    clade_left = clade_time_msprime(tree, tree.left_child(root))
    clade_right = clade_time_msprime(tree, tree.right_child(root))
    new_clade = [(clade_left[0][0] + clade_right[0][0], {"time": tree.time(root), "node": root})]
    return new_clade + clade_right + clade_left


def partition_dict_msprime(ts):
    hierarchie = {}
    for tree in ts.trees():
        left, right = tree.interval.left, tree.interval.right
        clade = {tuple(sorted(key)): value for key, value in clade_time_msprime(tree, tree.root) if value["time"] > 0}
        for partition in clade:
            if partition in hierarchie:
                if left == hierarchie[partition][-1]["right"] and hierarchie[partition][-1]["node"] == clade[partition]["node"]:
                    hierarchie[partition][-1].update({"right": right})
                else:
                    clade[partition].update({"left": left, "right": right})
                    hierarchie[partition].append(clade[partition])
            else:
                clade[partition].update({"left": left, "right": right})
                hierarchie[partition] = [clade[partition]]
    return hierarchie


def partition_dict_tsinfer(variants):
    ts, _, _, _ = test_tsinfer(variants, int(1e8), True)
    ts = td.date(ts, 1)
    dict_tsinfer = partition_dict_msprime(ts)
    return dict_tsinfer


def partition_dict_estimate(variants, crown, sample_size):
    partition_dict = partition_position(variants, sample_size)
    dict_depth_size = {}
    for partition in partition_dict:
        borns, depths = depths_clade(partition, partition_dict, variants, crown)
        for depth, born in zip(depths, borns):
            if partition in dict_depth_size:
                dict_depth_size[partition].append({"time": depth, "right": variants[born[1]].site.position, "left":  variants[born[0]].site.position})
            else:
                dict_depth_size[partition] = [{"time": depth, "right":  variants[born[1]].site.position, "left":  variants[born[0]].site.position}]
    return dict_depth_size


# def depht_hierarchie(crown, dict_depth_size, variants, sample_size):
#     partition_dict = partition_position(variants, sample_size)
#     print("b", len(variants))
#     for partition in dict_depth_size:
#         for dictionary in dict_depth_size[partition]:
#             inf, sup = dictionary["left"], dictionary["right"]
#             dictionary.update({"time": crown_stem(crown, inf, sup, variants, partition, partition_dict[partition]),
#                                "left": variants[inf].site.position,
#                                "right": variants[sup].site.position})
#     return dict_depth_size
#
#
# def partition_dict_hierarchie(variants, crown, sample_size):
#     #variants.append(Variant_m(1e8))
#     list_blocks = [(0, 0)] + detect_events(variants, sample_size)
#     list_blocks.append((len(variants) - 1, len(variants) - 1))
#     dict_depth_size = {}
#     individuals = np.array(range(sample_size))
#     print([(list_blocks[i][0] + 1, list_blocks[i + 1][1]) for i in range(len(list_blocks) - 1)])
#     for inf, sup in [(list_blocks[i][0] + 1, list_blocks[i + 1][1]) for i in range(len(list_blocks) - 1)]:
#         for variant in variants[inf:sup + 1]:
#             partition = tuple(individuals[variant.genotypes == 1])
#             if partition in dict_depth_size:
#                 if inf <= dict_depth_size[partition][-1]["right"] + 1:
#                     dict_depth_size[partition][-1].update({"right": sup})
#                 else:
#                     dict_depth_size[partition].append({"left": inf, "right": sup})
#             elif len(partition) > 1:
#                 dict_depth_size[partition] = [{"left": inf, "right": sup}]
#     for key in dict_depth_size:
#         for elem in dict_depth_size[key]:
#             print(elem)
#     return depht_hierarchie(crown, dict_depth_size, variants, sample_size)


def depht_hierarchie(crown, dict_depth_size, variants, sample_size):
    partition_dict = partition_position(variants, sample_size)
    for partition in dict_depth_size:
        for dictionary in dict_depth_size[partition]:
            inf, sup = dictionary["left"], dictionary["right"]
            dictionary.update({"time": crown_stem(crown, inf, sup, variants, partition, partition_dict[partition]),
                               "left": variants[inf].site.position,
                               "right": variants[sup].site.position})
    return dict_depth_size


def partition_dict_hierarchie(variants, crown, sample_size):
    list_blocks = [(0, 0)] + detect_events(variants, sample_size)
    list_blocks.append((len(list_blocks) - 1, len(list_blocks) - 1))
    dict_depth_size = {}
    individuals = np.array(range(sample_size))
    for inf, sup in [(list_blocks[i][0], list_blocks[i + 1][1]) for i in range(len(list_blocks) - 1)]:
        for variant in variants[inf:sup]:
            partition = tuple(individuals[variant.genotypes == 1])
            if partition in dict_depth_size:
                if inf <= dict_depth_size[partition][-1]["right"] + 1:
                    dict_depth_size[partition][-1].update({"right": sup})
                else:
                    dict_depth_size[partition].append({"left": inf, "right": sup})
            elif len(partition) > 1:
                dict_depth_size[partition] = [{"left": inf, "right": sup}]
    return depht_hierarchie(crown, dict_depth_size, variants, sample_size)


def partition_estimate_msprime(list_estimate, list_msprime, mu=1):
    estimator, ms = [], []
    index = 0
    for estimate in list_estimate:
        while index < len(list_msprime):
            mspr = list_msprime[index]
            if mspr["right"] < estimate["right"] and mspr["left"] > estimate["left"]:
                estimator = np.append(estimator, estimate["time"] / mu)
                ms = np.append(ms, mspr["time"])
            if mspr["right"] > estimate["right"]:
                break
            index += 1
    return estimator, ms


def estimate_v_msprime(estimate_msprime, dict_estimate, dict_msprime, mu=1):
    for partition in dict_estimate:
        if len(partition) < 6:
            estimate, ms = partition_estimate_msprime(dict_estimate[partition], dict_msprime[partition], mu)
            estimate_msprime[len(partition)][0] = np.concatenate([estimate_msprime[len(partition)][0], ms])
            estimate_msprime[len(partition)][1] = np.concatenate([estimate_msprime[len(partition)][1], estimate])
    return estimate_msprime


def randomi(dict_times, dictio, mu=1):
    #dict_rimes = {key: [] for key in range(2, 7)}
    for key in dictio:
        dict_times[len(key)] += [dictioi["time"] / mu for dictioi in dictio[key] if dictioi["time"] > 0]
    return dict_times


def randomi_2(list_times, dictio, mu=1):
    for key in dictio:
        list_times += [dictioi["time"] / mu for dictioi in dictio[key] if dictioi["time"] > 0]
    return list_times


def replicate_ems(params, nb):
    dict_estimate = {}
    dict_ms = {}
    for mu_ro in [1, 10, 100]:
        estimate_msprime = dict([(i + 1, [np.array([]), np.array([])]) for i in range(1, params["sample_size"] - 1)])
        params.update({"mu": mu_ro * 8e-6})
        dict_estimate[mu_ro], dict_ms[mu_ro] = {key: [] for key in range(2, 7)}, {key: [] for key in range(2, 7)}
        for i in range(nb):
            ts, _, _, variants = msprime_simulate_variants(params)
            a = partition_dict_msprime(ts)
            b = partition_dict_hierarchie(variants, True, 6)
            #b = partition_dict_tsinfer(variants, params["mu"])
            dict_estimate[mu_ro] = randomi(dict_estimate[mu_ro], b, params["mu"]) #estimate_msprime
            dict_ms[mu_ro] = randomi(dict_ms[mu_ro], a)
            estimate_msprime = estimate_v_msprime(estimate_msprime, b, a, params["mu"])
            cumulative_times_size(dict_estimate[mu_ro], dict_ms[mu_ro], mu_ro, "fig_dir")
        msprime_estimate_scatters(estimate_msprime, mu_ro, "fig_dir", params)
    cumulative_times_size_2(dict_estimate, dict_ms, "fig_dir")


def replicate_ems_scenario(params, nb):
    dict_estimate = {}
    dict_ms = {}
    for kappa in [0.1, 1, 10]:
        estimate_msprime = dict([(i + 1, [np.array([]), np.array([])]) for i in range(1, params["sample_size"] - 1)])
        params.update({"Kappa": kappa})
        dict_estimate[kappa], dict_ms[kappa] = {key: [] for key in range(2, 7)}, {key: [] for key in range(2, 7)}
        for i in range(nb):
            ts, _, _, variants = msprime_simulate_variants(params)
            a = partition_dict_msprime(ts)
            b = partition_dict_estimate(variants, True, 6)
            estimate_msprime = estimate_v_msprime(estimate_msprime, b, a, params["mu"])
            dict_estimate[kappa] = randomi(dict_estimate[kappa], b, params["mu"])  # estimate_msprime
            dict_ms[kappa] = randomi(dict_ms[kappa], a)
            #estimate_msprime = estimate_v_msprime(estimate_msprime, b, a, params["mu"])
        msprime_estimate_scatters(estimate_msprime, kappa, "fig_dir/scenario/", params)
        #cumulative_times_size(dict_estimate[kappa], kappa, "fig_dir/scenario/")
    cumulative_times_size_3(dict_estimate, dict_ms, "fig_dir/scenario/")


def comp_method(dict_method, params, nb):
    ts, _, _, variants = msprime_simulate_variants(params)
    dict_method_r = {}
    for key in dict_method:
        dict_rsquared = {i: [] for i in range(2, params["sample_size"])}
        for i in range(nb):
            estimate_msprime = dict([(i + 1, [np.array([]), np.array([])]) for i in range(1, params["sample_size"] - 1)])
            ts, _, _, variants = msprime_simulate_variants(params)
            estimate_msprime = estimate_v_msprime(estimate_msprime, dict_method[key](variants, True, 6),
                                                  partition_dict_msprime(ts), params["mu"])
            dict_rsquared, _ = rsquared(estimate_msprime, dict_rsquared)
        dict_method_r[key] = dict_rsquared
    return dict_method_r


########################################################################################################################
##########################################plot###########################################################
########################################################################################################################


def msprime_estimate_scatters(dict_times, mu_ro, dir_fig, params):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    dict_rsquared = {i: [] for i in range(2, params["sample_size"])}
    dict_reg = {i: [] for i in range(2, params["sample_size"])}
    dict_rsquared, dict_reg = rsquared(dict_times, dict_rsquared, dict_reg)
    for key in dict_times:
        data = dict_times[key]
        axs[(key - 2) // 2, (key - 2) % 2].plot(data[0], data[0], color='blue', ls="-")
        axs[(key - 2) // 2, (key - 2) % 2].plot(data[0], dict_reg[key], color='red', ls="-")
        axs[(key - 2) // 2, (key - 2) % 2].scatter(data[0], data[1], alpha=0.1)
        #axs[(key - 2) // 2, (key - 2) % 2].set_title(letters[][])
        axs[(key - 2) // 2, (key - 2) % 2].text(0.4, 0.8, f'k={key}, r2={round(dict_rsquared[key][0], 2)}', horizontalalignment='center', verticalalignment='center',
                      transform=axs[(key - 2) // 2, (key - 2) % 2].transAxes, fontsize=16)
    for ax in axs.flat:
        ax.set(xlabel='temps msprime', ylabel='temps estimés')
    for ax in axs.flat:
        ax.label_outer()
    fig.savefig(f"{dir_fig}/scatter_{mu_ro}.png", dpi=200)


def rsquared_box(dict_squared, mu_ro, dir_fig, sample_size):
    fig, axs = plt.subplots(2, 2)
    labels = dict_squared.keys()
    for effective in range(2, sample_size):
        data = []
        for key in dict_squared:
            data.append(dict_squared[key][effective])
        axs[(effective - 2) // 2, (effective - 2) % 2].boxplot(data)
        #axs[(effective - 2) // 2, (effective - 2) % 2].set_xticks([1, 2, 3], labels)from
        for ax in axs.flat:
            ax.set(xlabel='msprime time', ylabel='estimate times')
        for ax in axs.flat:
            ax.label_outer()
    fig.savefig(f"{dir_fig}/rsquared_{mu_ro}.png", dpi=200)


# def cumulative_times(dict_ms, dict_estimate, mu_ro, dir_fig):
#     fig, axs = plt.subplots()
#     size = 2
#     plt.plot(np.arange(0, 7, 0.1),  -np.exp(-2 * np.arange(0, 7, 0.1)) + 1, ls="-", color="black")
#     plt.plot(np.arange(0, 7, 0.1), -np.exp( - np.arange(0, 7, 0.1)) + 1, ls="-", color="black")
#     colors = ["red", "blue", "green"]
#     i = 0
#     for key in dict_estimate:
#         data = sorted(dict_ms[key][size])
#         data_2 = sorted(dict_estimate[key][size])
#         plt.plot(data, np.arange(0, len(data),  1) / len(data), ls="-", color=colors[i])
#         plt.plot(data_2, np.arange(0, len(data_2),  1) / len(data_2), ls="--", color=colors[i])
#         i += 1
#     plt.xlabel("temps de coalescence")
#     # plt.xlim((0, 5))
#     # plt.legend(["theoric"] + list(dict_times.keys()))
#     fig.savefig(f"{dir_fig}/cumulative2_{mu_ro}.png", dpi=200)

def cumulative_times(dict_ms, dict_estimate, dir_fig):
    fig, axs = plt.subplots(1, 2)
    mns, stds = [], []
    colors = ["red", "blue", "green"]
    i = 0
    for key in dict_estimate:
        data, mn = np.array(sorted(dict_ms[key])), np.mean(dict_ms[key])
        data_2, mn_2 = np.array(sorted(dict_estimate[key])), np.mean(dict_estimate[key])
        axs[0].plot(data / mn, np.arange(0, len(data),  1) / len(data), ls="-", color=colors[i])
        axs[0].plot(data_2 / mn_2, np.arange(0, len(data_2),  1) / len(data_2), ls="--", color=colors[i])
        mns.append(mn_2)
        stds.append(np.std(dict_estimate[key]))
        i += 1
    axs[0].set_xlabel("temps de coalescence")
    axs[1].bar(range(3), mns, yerr=stds)
    axs[0].set_xlim((0, 5))
    # plt.legend(["theoric"] + list(dict_times.keys()))
    fig.savefig(f"{dir_fig}/cumulative2.png", dpi=200)


def cumulative_times_size(dict_times, dict_estimate, mu_ro, dir_fig):
    print(mu_ro)
    fig, axs = plt.subplots()
    for size in dict_times:
        if size < 6:
            plt.plot(sorted(dict_times[size]), np.arange(0, len(dict_times[size]),  1) / len(dict_times[size]), ls="-")
            plt.plot(sorted(dict_estimate[size]), np.arange(0, len(dict_estimate[size]), 1) / len(dict_estimate[size]), ls="--", color="black")
    # plt.legend(list(dict_times.keys()))
    plt.xlabel("temps de coalescence")
    plt.xlim((0, 5))
    fig.savefig(f"{dir_fig}/cumulative_size_{mu_ro}.png", dpi=200)


def cumulative_times_size_2(dict_times_sizes, dict_estimate_sizes, dir_fig):
    fig, axs = plt.subplots(3, 1, figsize=(6, 15), sharex=True)
    i = 0
    letters = ["A", "B", "C"]
    for mu_ro in dict_times_sizes:
        dict_times, dict_estimate = dict_times_sizes[mu_ro], dict_estimate_sizes[mu_ro]
        print(dict_times, dict_estimate)
        for size in dict_times:
            print(size)
            if size < 6:
                axs[i].plot(sorted(dict_estimate[size]),
                            np.arange(0, len(dict_estimate[size]), 1) / len(dict_estimate[size]), ls="--",
                            color="black")
                axs[i].plot(sorted(dict_times[size]), np.arange(0, len(dict_times[size]),  1) / len(dict_times[size]), ls="-")
                axs[i].text(0.1, 0.9, letters[i], horizontalalignment='center', verticalalignment='center',
                         transform=axs[i].transAxes)
        #axs[i // 2, i % 2].legend(list(dict_times.keys()))
        axs[i].set_xlim((0, 5))
        i += 1
    axs[2].set_xlabel("temps de coalescence", fontsize=15)
    fig.savefig(f"{dir_fig}/cumulative_size.png", dpi=200)


def cumulative_times_size_3(dict_times_sizes, dict_estimate_sizes, dir_fig):
    fig, axs = plt.subplots(3, 1, figsize=(6, 15))
    i = 0
    letters = ["A", "B", "C"]
    for mu_ro in dict_times_sizes:
        dict_times, dict_estimate = dict_times_sizes[mu_ro], dict_estimate_sizes[mu_ro]
        for size in dict_times:
            if size < 6:
                axs[i].plot(sorted(dict_estimate[size]),
                            np.arange(0, len(dict_estimate[size]), 1) / len(dict_estimate[size]), ls="--",
                            color="black")
                axs[i].plot(sorted(dict_times[size]), np.arange(0, len(dict_times[size]),  1) / len(dict_times[size]), ls="-")
                axs[i].text(0.1, 0.9, letters[i], horizontalalignment='center', verticalalignment='center',
                         transform=axs[i].transAxes)
                axs[i].set_xlim((0, 5))
        #axs[i // 2, i % 2].legend(list(dict_times.keys()))
        i += 1
    axs[2].set_xlabel("temps de coalescence")
    fig.savefig(f"{dir_fig}/cumulative_size_s.png", dpi=200)


def residues_box(dict_times, mu_ro, dir_fig, sample_size):
    fig, axs = plt.subplots(2, 2)
    for effective in range(2, sample_size):
        data = []
        for key in dict_times:
            data.append((dict_times[key][effective][1] - dict_times[key][effective][0]) / dict_times[key][effective][0])
        axs[(effective - 2) // 2, (effective - 2) % 2].boxplot(data)
        #axs[(effective - 2) // 2, (effective - 2) % 2].set_xticks([1, 2, 3])
        for ax in axs.flat:
            ax.set(xlabel='msprime time', ylabel='estimate times')
        for ax in axs.flat:
            ax.label_outer()
    fig.savefig(f"{dir_fig}/sbox_residue_{mu_ro}.png", dpi=200)


def main():
    params = {"sample_size": 6, "Ne": 1, "ro": 8e-6, "mu": 8e-5, "Tau": 1, "Kappa": 1, "length": int(1e8)}
    out_csv, out_fig = "csv_dir", "fig_dir"
    # ts, _, _, variants = msprime_simulate_variants(params)
    # partition_dict_tsinfer(variants, params["mu"])
    # b = partition_dict_hierarchie(variants, True, 6)
    # print(len(list(b.keys())))
    # b2 = partition_dict_estimate(variants, True, 6)
    # print(len(list(b.keys())))
    # count = 0
    # for key in b:
    #     count += len(b[key])
    # print(count)
    # count = 0
    # for key in b2:
    #     count += len(b2[key])
    # print(count)
    replicate_ems(params, 5)
    #replicate_ems_scenario(params, 1)
    #rsquared_box(comp_method({"hierarchies": partition_dict_hierarchie, "estimate": partition_dict_estimate}, params, 30), 1, out_fig, 6)


if __name__ == "__main__":
    main()

