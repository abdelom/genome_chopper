import numpy as np
import pandas as pd
import collections as cl
import matplotlib.pyplot as plt
from classification_breakpoints import *
from msprime_simulation import msprime_simulate_variants, test_tsinfer
import time

"""
implémentation de deux algorithmes de détection de point de recombinaison dans un alignement de séquences apparentées et comparaison entre elles et
tsinfer
"""


########################################################################################################################
##########################################algo naif####################################################################
########################################################################################################################


def internal_incompatibility(genotype1, genotype2, oriented=True):
    """
    four gamete test enlarged to discrete recombnation events since genotypes
    are oriented ( 0 = ancestral state, 1 = derivative state)
    two genotypes are compatible, there is no breakpoint between them,
    when they exlude each others or one of them include the other
    parameters:
        genotype1, ganotype2, array of int values taken only 0 or 1 as value
    return:
        boolean, True if the two sites if there is an incompatible or discrete
        recombination event between them, then False
    """
    if not oriented:
        return len(set([*zip(genotype1, genotype2)])) == 4
    set_genotype_1 = {i for i in range(len(genotype1)) if genotype1[i]}
    set_genotype_2 = {i for i in range(len(genotype2)) if genotype2[i]}
    return not (
            not set_genotype_1.intersection(set_genotype_2)
            or set_genotype_1.union(set_genotype_2) == set_genotype_1
            or set_genotype_1.union(set_genotype_2) == set_genotype_2
    )


def shortening(list_incompatible):
    """
    parameters:
    list_incompatible, list of int's tuples, each tuple define two incompatible variant
    sites defining a breakpoint
    return:
    list_incompatible, list of non overlapping int's tuples
    """
    list_incompatible = sorted(list_incompatible, key=lambda t: (t[0], t[1]))
    index = len(list_incompatible) - 1
    while index > 0:
        min1, max1 = list_incompatible[index - 1]
        min2, max2 = list_incompatible[index]
        if min1 <= min2 and max1 >= max2:  # if the second tuple is emcompassed within the first
            list_incompatible[index - 1] = list_incompatible[index]
            del list_incompatible[index]
        elif (min1 <= min2 < max1) and max1 < max2:  # if the two tuples are overlapping
            del list_incompatible[index]
            list_incompatible[index - 1] = (min2, max1)
        index -= 1
    return list_incompatible


def choping(mld, variants, length):
    """
    entrée:
        mld: liste de tuples définissant un point de recombinaisons
        variants: liste d'objet Variants du module msprime
        length: longueur de l'alignement
    sortie:
        liste de float: position des points de recombinaisons sur l'alignement
    """
    return [0.0] + [(variants[end].site.position + variants[start].site.position) / 2
                    for start, end in mld] + [length]


def detect_internal_incompatibilities(variants, oriented=True, thresold=20):
    """
    detection of the incomaptible paires of variant sites between two born considering
    only sequences displaying derivative state on the segregative site.
    paramters:
        variants, list of msprime Variant class's instances,
        segregated, boolean list, true if, for the segragarive site, the corresponding
        sequence display the derivative state.
        born_inf, int: the closest site incompatible site with the segregative_site
        before him
        born_sup, int: the closest site incompatible site with the segregative_site
        after
        thresold, int:
    """
    list_incompatible_sites = []
    for i in range(len(variants)):
        j = i + 1
        while j < i + thresold and j < len(variants):
            if internal_incompatibility(variants[i].genotypes,
                                        variants[j].genotypes, oriented):
                list_incompatible_sites.append((i, j))
            j += 1
    return shortening(list_incompatible_sites)


########################################################################################################################
##########################################algo underachieve############################################################
########################################################################################################################


def internal_incompatibility_2(set_genotype_1, set_genotype_2):
    """
    four gamete test enlarged to discrete recombnation events since genotypes
    are oriented ( 0 = ancestral state, 1 = derivative state)
    two genotypes are compatible, there is no breakpoint between them,
    when they exlude each others or one of them include the other
    parameters:
        genotype1, ganotype2, array of int values taken only 0 or 1 as value
    return:
        boolean, True if the two sites if there is an incompatible or discrete
        recombination event between them, then False
    """
    return not (
            not set_genotype_1.intersection(set_genotype_2)
            or set_genotype_1.union(set_genotype_2) == set_genotype_1
            or set_genotype_1.union(set_genotype_2) == set_genotype_2
    )


def closest_incompatibility(index: object, sub_set: object) -> object:
    """
    :param index: dcitionnair, les clé sont les tuples chacun associé à une bipartition corespondant à un site polymorche déjà
    rencontré, la valeur à la position du derniers snp, générant cette bipartition rencontré
    :param sub_set: bipartition di site polymorphe courant
    :return: la position du site polymorphe incompatible le plus proche du site courant, -1 s'il n'y a pas d'incompatibilité
    """
    position = [-1]
    for sub_set_tmp in index:
        if internal_incompatibility_2(set(sub_set), set(sub_set_tmp)):
            position.append(index[sub_set_tmp])
    return max(position)  # -1 le site est compatibles avec toute les partiotn précédeme,nt rencontrée


def built_index(start, max_start, variants, individuals):
    """

    :param start:
    :param max_start:
    :param variants:
    :param individuals:
    :return:
    """
    block_start, position = start, start
    index = {tuple(individuals[variants[start].genotypes == 1]): start}
    while position < len(variants):
        variant = variants[position]
        sub_set = tuple(individuals[variant.genotypes == 1])
        if len(sub_set) != 1:
            if sub_set not in index:
                position_2 = closest_incompatibility(index, sub_set)
                if position_2 != -1:
                    if position_2 > max_start:
                        return block_start, position_2, position
                    block_start, position = position_2 + 1, position_2 + 1
                    sub_set = tuple(individuals[variants[position].genotypes == 1])
                    index = {sub_set: position}
                    position += 1
                    continue
            index[sub_set] = position
        position += 1
    return block_start, - 1, len(variants) - 1


# def detect_events(variants, nb):
#     """
#
#     :param variants: liste d'objet Variants du module msprime
#     :param nb: nombre de séquence dans l'alignement
#     :return: liste de tuples délimitants des bloques de dans l'alignement de séquence sans incompatibilitées
#     ces blocs peuvent être chevauchant
#     """
#     individuals = np.array((range(nb)))
#     block_start, start, block_end = built_index(0, 0, variants, individuals)
#     list_block = [(block_start, block_end)]
#     while block_end < len(variants) - 1:
#         block_start, start, block_end = built_index(start + 1, block_end, variants, individuals)
#         list_block.append((block_start - 1, block_end))
#     return [(list_block[i + 1][0], list_block[i][1]) for i in range(len(list_block) - 1)]  # list_block


def detect_events(variants, nb):
    """
    :param variants: liste d'objet Variants du module msprime
    :param nb: nombre de séquence dans l'alignement
    :return: liste de tuples délimitants des bloques de dans l'alignement de séquence sans incompatibilitées
    ces blocs peuvent être chevauchant
    """
    individuals = np.array((range(nb)))
    block_start, start, block_end = built_index(0, 0, variants, individuals)
    list_block = [(block_start, block_end)]
    while block_end < len(variants) - 1:
        block_start, start, block_end = built_index(start + 1, block_end, variants, individuals)
        list_block.append((block_start - 1, block_end))
    return [(list_block[i + 1][0] + 1, list_block[i][1]) for i in range(len(list_block) - 1)]  # list_block


#######################################################################################################################
############################################checked####################################################################
#######################################################################################################################


# def incompatibility_in_block(partitions):
#     for index, partition in enumerate(partitions):
#         for jindex in range(index + 1, len(partitions)):
#             if internal_incompatibility_2(set(partition), set(partitions[jindex])):
#                 return True
#     return False


# def checked_incompatibilities(list_blocks, variants, nb):
#     count = 0
#     individuals = np.array((range(nb)))
#     for inf, sup in list_blocks:
#         dict_partition = {}
#         for index, variant in enumerate(variants[inf + 1:sup]):
#             partition = tuple(individuals[variant.genotypes == 1])
#             if len(partition) == 1:
#                 continue
#             if partition not in dict_partition:
#                 dict_partition[partition] = None
#         if incompatibility_in_block(list(dict_partition.keys())):
#             count += 1
#     return count


# def hierarchie(variants, inf, sup, individuals):
#     hierarchie = {}
#     for variant in variants[inf: sup]:
#         index = tuple(individuals[variant.genotypes == 1])
#         if index not in hierarchie:
#             hierarchie[index] = None
#     return hierarchie


# def compar_hierarchie(hierarchie_1, hierarchie_2):
#     for index in hierarchie_1:
#         if index in hierarchie_2:
#             continue
#         for index_2 in hierarchie_2:
#             if internal_incompatibility_2(set(index), set(index_2)):
#                 return True
#     return False


# def check_blocks(blocks, variants, sample_size):
#     individuals = np.array(range(sample_size))
#     inf, sup = blocks[0]
#     count = 0
#     hierarchie_1 = hierarchie(variants, inf, sup, individuals)
#     for inf, sup in blocks[1:]:
#         hierarchie_2 = hierarchie(variants, inf, sup, individuals)
#         if not compar_hierarchie(hierarchie_1, hierarchie_2):
#             count += 1
#             # print("a")
#             # print(hierarchie_1.keys())
#             # print(hierarchie_2.keys())
#         hierarchie_1 = hierarchie_2
#     print(count)


########################################################################################################################
########################################## comparaison##################################################################
########################################################################################################################


def closest(obs_events, th_events, variants, length, chop=True):
    """
    retourne la moyennes des distances entre les événeents de recombinaisons et l'événement de recombinaison rélle le plus proche
    """
    positions, kind = zip(*obs_events)
    obs_events = np.array(positions)[np.array(kind) != "silent"]
    obs_events = [-length] + list(obs_events) + [2 * length]
    if chop:
        th_events = choping(th_events, variants, length)[1:-1]
    index = 0
    list_dist = []
    left = obs_events[0]
    for right in obs_events[1:]:
        while index < len(th_events) and left <= th_events[index] <= right:
            list_dist.append(min(th_events[index] - left, right - th_events[index]))
            index += 1
        left = right
    return list_dist


def th_events_ts_infer(obs_events, variants, params):
    """
    retourne le temps de calcul de tsinfer, le niombre de changements de topologies detecté et la distance au point de recombinaison
    discret ou incompatible le plus proche.
    """
    start = time.time()
    ts, edges, th_events, variants = test_tsinfer(variants, params["length"], simplify=True)
    end = time.time() - start
    dist_closest = closest(obs_events, th_events, variants, params["length"], chop=False)
    print(end)
    return [cl.Counter(list(zip(*class_brkpoints(edges, th_events, unroot=False)))[1])["incompatible"], \
            end, dist_closest]


def th_events_ag(obs_events, variants, params):
    """
    retourne le temps de calcul de tsinfer, le niombre de changements de topologies detecté et la distance au point de recombinaison
    discret ou incompatible le plus proche.
    """
    start = time.time()
    th_events = detect_events(variants, params["sample_size"])
    end = time.time() - start
    dist_closest = closest(obs_events, th_events.copy(), variants, params["length"])
    th_events.append((len(variants), len(variants)))
    th_events = [(-1, -1)] + th_events
    print(end)
    return len(th_events), end, dist_closest


def th_events_ek(obs_events, variants, params):
    """
    retourne le temps de calcul de la méhde naïve, le niombre de changements de topologies detecté et la distance au point de recombinaison
    discret ou incompatible le plus proche.
    """
    start = time.time()
    th_events = detect_internal_incompatibilities(variants, True, params["thresold"])
    end = time.time() - start
    dist_closest = closest(obs_events, th_events.copy(), variants, params["length"])
    th_events.append((len(variants), len(variants)))
    th_events = [(-1, -1)] + th_events
    return len(th_events), end, dist_closest


def data_simulation(params):
    """
    params: paramètres du sénario démographique
    simule un alignemùent msprime et teste les différentes méthodes de détections des point de recombinaisons
    """
    closest_dist, result, list_time = [], [], []
    thresolds = [0]
    ts, edges, events, variants = msprime_simulate_variants(params)
    print(len(variants))
    events = class_brkpoints(edges, events, unroot=True)
    events_count = cl.Counter(list(zip(*events))[1])
    total = events_count["incompatible"] + events_count["discret"]
    result.append(total)
    for func in [th_events_ts_infer, th_events_ag, th_events_ek]:
        if func == th_events_ek:
            thresolds = (50, 100)
        for thr in thresolds:
            params.update({"thresold": thr})
            nb_events, end, dist_closest = func(events, variants, params)
            result, list_time = result + [nb_events], list_time + [end]
            closest_dist.append(dist_closest)
    return result, list_time, closest_dist


def format_closest(dataf, method):
    list_value, list_method = [], []
    for index, liste in enumerate(dataf):
        list_value += liste
        list_method += [method[index]] * len(liste)
    return pd.DataFrame(zip(list_method, list_value), columns=["method", "value"])


def method_comparaison(params, nb_repetition, mu_ro, out_dir):
    i = 0
    list_times, list_results, data_closest = [], [], pd.DataFrame(columns=["method", "value"])
    list_result = []
    while i < nb_repetition:
        result, list_time, nearest = data_simulation(params)
        print(result, list_time)
        list_times.append(list_time.copy())
        list_result.append(result.copy())
        data_closest = pd.concat((data_closest, format_closest(nearest, ["tsinfer", "hierarchie", "naive_50", "naive_100"])))
        i += 1
    # pd.DataFrame(list_times, columns=["tsinfer", "hierarchie", "naive_50", "naive_100"]).to_csv(
    #     f"{out_dir}/time_{mu_ro}")
    # pd.DataFrame(
    #     list_result, columns=["total", "tsinfer", "hierarchie", "naive_50", "naive_100"]).to_csv(
    #     f"{out_dir}/result_{mu_ro}")
    # data_closest.to_csv(
    #     f"{out_dir}/closest_dist_{mu_ro}")


########################################################################################################################
########################################## Plot#########################################################################
########################################################################################################################

def plot_time_error():
    leg = ["tsinfer", "hierarchie", "naive_50", "naive_100"]
    data_mean = pd.DataFrame(columns=leg)
    data_std = pd.DataFrame(columns=leg)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    l_mu_ro = [1, 10, 100]
    for mu_ro in [1, 10, 100]:
        dataf = pd.read_csv(f"csv_dir/time_{mu_ro}", index_col=0)
        dataf.columns = leg
        data_mean = pd.concat((data_mean, pd.DataFrame(np.log10(dataf.mean(axis=0))).transpose()))
        #data_std = pd.concat((data_std, pd.DataFrame(np.log10(dataf.std(axis=0))).transpose()))
    leg = ["tsinfer", "hierarchie", "naive_100"]
    for elem in leg:
        #print(data_mean[elem])
        plt.plot(range(3), data_mean[elem], linestyle='--', marker='o')  #, data_std[elem]
        plt.legend(leg, title="Legend")
        plt.ylabel("temps (en seconde)", fontsize=12)
        plt.xticks(range(3), [1, 10, 100])
        plt.yticks(range(-4, 2), ["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$"])
        plt.xlabel("mu / rho",  fontsize=12)
    fig.savefig(f"fig_dir/time_error_2.png", dpi=100)


def plot_closest_error():
    leg = ["tsinfer", "hierarchie", "naive_50", "naive_100"]
    data_mean = pd.DataFrame(columns=leg)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for mu_ro in [1, 10, 100]:
        dataf = pd.read_csv(f"csv_dir/closest_{mu_ro}", index_col=0)
        dataf.columns = leg
        data_mean = pd.concat((data_mean, pd.DataFrame(np.log10(dataf.mean(axis=0))).transpose()))
    leg = ["tsinfer", "hierarchie", "naive_100"]
    for elem in leg:
        plt.plot(range(3), data_mean[elem], linestyle='--', marker='o')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.ylabel("distance en paire de bases", fontsize=16)
    plt.xlabel('mu / rho', fontsize=16)
    plt.xticks(range(3), ["1", "10", "100"])
    plt.yticks(np.arange(2.5, 4.1, 0.5), ["$10^{2.5}$", "$10^{3}$", "$10^{3.5}$", "$10^{4}$"])
    fig.savefig(f"fig_dir/closes.png", dpi=100)


def nb_event():
    leg = ["tsinfer", "hierarchie", "naive_50", "naive_100"]
    data_mean = pd.DataFrame(columns=leg)
    data_std = pd.DataFrame(columns=leg)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for mu_ro in [1, 10, 100]:
        data = pd.read_csv(f"csv_dir/result_{mu_ro}", index_col=0)
        data.columns = ["total"] + leg
        data_mean = pd.concat((data_mean, pd.DataFrame(data.mean(axis=0)).transpose()))
        data_std = pd.concat((data_std, pd.DataFrame(data.std(axis=0)).transpose()))
    leg = ["tsinfer", "hierarchie", "naive_100"]
    for elem in leg:
        #plt.plot(range(3), data_mean[elem], linestyle='--', marker='o')
        plt.errorbar(range(3), data_mean[elem], data_std[elem], marker="o", linestyle='--')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.axhline(y=np.mean(data["total"]))
    # plt.axhline(y=np.mean(data["total"]) - np.std(data["total"]), color="r", linestyle='--')
    # plt.axhline(y=np.mean(data["total"]) + np.std(data["total"]), color="r", linestyle='--')
    plt.ylabel("nombre d'événements detectés", fontsize=16)
    plt.xlabel('mu / rho', fontsize=16)
    plt.xticks(range(3), ["1", "10", "100"])
    #plt.yticks(np.arange(2.5, 4.1, 0.5), ["$10^{2.5}$", "$10^{3}$", "$10^{3.5}$", "$10^{4}$"])
    fig.savefig(f"fig_dir/resul.png", dpi=100)


def list_closest(data, methods):
    list_data = []
    for method in methods:
        list_data.append(data["value"][data["method"] == method])
    return list_data


def closest_panel(out_csv):
    leg = ["tsinfer", "hierarchie", "naive_50", "naive_100"]
    data_mean = pd.DataFrame(columns=leg)
    data_std = pd.DataFrame(columns=leg)
    fig, ax = plt.subplots(2, 2,  figsize = (15, 10))
    letters = [["A", "B"], ["C", "D"]]
    markers = [["o", "+"],["square", "diamond"]]
    mu_ro = 1
    for i in range(2):
        for j in range(2):
            for label in (ax[i, j].get_xticklabels() + ax[i, j].get_yticklabels()):
                label.set_fontsize(16)
            if mu_ro != 1000:
                if j == 0:
                    ax[i, j].set_ylabel("distance à l'événément\nle plus proche", fontsize=12)
                ax[i, j].set_xlabel('méthode', fontsize=16)
                data = pd.read_csv(f"{out_csv}/closest_dist_{mu_ro}", index_col=0)
                ax[i, j].boxplot(list_closest(data, leg), patch_artist=True)
                ax[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                #ax[i, j].set_xticks([1, 2, 3, 4], leg)
                ax[i, j].text(0.1, 0.9, letters[i][j], horizontalalignment='center', verticalalignment='center',
                            transform=ax[i, j].transAxes, fontsize=20)
                data_mean = pd.concat((data_mean, pd.DataFrame(np.log10(1 + data.groupby(['method']).mean())).transpose()))
                data_std = pd.concat((data_std, pd.DataFrame(np.log10(data.groupby(['method']).std())).transpose()))
            mu_ro = mu_ro * 10
    for elem in leg:
        ax[1, 1].plot(range(3), data_mean[elem], linestyle='--', marker='o')
        ax[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[1, 1].set_xlabel("log(distance ", fontsize=12)
    ax[1, 1].set_xlabel('mu / rho', fontsize=16)
    ax[1, 1].set_xticks(range(3))
    ax[1, 1].set_xticklabels(["1", "10", "100"])
    ax[1, 1].set_yticks(np.arange(2.5, 4.1, 0.5))
    ax[1, 1].set_yticklabels(["$10^{2.5}$", "$10^{3}$", "$10^{3.5}$", "$10^{4}$"])
    ax[1, 1].text(0.1, 0.9, letters[i][j], horizontalalignment='center', verticalalignment='center',
                  transform=ax[i, j].transAxes, fontsize=20)
    fig.savefig(f"closest_panel.png", dpi=200)


def scatter_panel(out_csv):
    letters = [["A", "B"], ["C", "D"]]
    markers = ["o", "s", "D", "X"]
    leg = ["tsinfer", "hierarchie", "naive_50", "naive_100"]
    data_mean = pd.DataFrame(columns=leg)
    data_std = pd.DataFrame(columns=leg)
    fig, ax = plt.subplots(2, 2,  figsize = (15, 10))
    mu_ro = 1
    for i in range(2):
        for j in range(2):
            for label in (ax[i, j].get_xticklabels() + ax[i, j].get_yticklabels()):
                label.set_fontsize(16)
            if mu_ro != 1000:
                if j == 0:
                    ax[i, j].set_ylabel('événements détectés (x100)', fontsize=16)
                ax[i, j].set_xlabel('événements msprime (x100)', fontsize=16)
                data = pd.read_csv(f"{out_csv}/result_{mu_ro}", index_col=0)
                data.columns = ['total'] + leg
                ax[i, j].plot(data['total'], data['total'], ls="-")
                ax[i, j].scatter(data['total'], data['tsinfer'], marker="o", alpha=0.3)
                ax[i, j].scatter(data['total'], data['hierarchie'], marker="s", alpha=0.3)
                ax[i, j].scatter(data['total'], data['naive_100'], marker="X", alpha=0.3)
                ax[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                ax[i, j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                ax[i, j].text(0.1, 0.9, letters[i][j], horizontalalignment='center', verticalalignment='center',
                              transform=ax[i, j].transAxes, fontsize=20)
                data_mean = pd.concat((data_mean, pd.DataFrame(data.mean(axis=0)).transpose()))
                data_std = pd.concat((data_std, pd.DataFrame(data.std(axis=0)).transpose()))
            mu_ro = mu_ro * 10
    for index, elem in enumerate(leg):
        ax[1, 1].errorbar(range(3), data_mean[elem], data_std[elem], marker="o", linestyle='--')
    print(data_mean)
    ax[1, 1].axhline(y=np.mean(data["total"]))
    ax[1, 1].axhline(y=np.mean(data["total"]) - np.std(data["total"]), color="r", linestyle='--')
    ax[1, 1].axhline(y=np.mean(data["total"]) + np.std(data["total"]), color="r", linestyle='--')
    ax[i, j].set_xlabel('mu / rho', fontsize=16)
    ax[i, j].set_xticks(range(3))
    ax[i, j].set_xticklabels(["1", "10", "100"])
    ax[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    #ax[i, j].ticklabel_format(axis="x", scilimits=(0, 0))
    ax[i, j].text(0.1, 0.9, letters[i][j], horizontalalignment='center', verticalalignment='center',
                  transform=ax[i, j].transAxes, fontsize=20)
    fig.savefig(f"scatter_panel.png", dpi=200)


def main():
    params = {"sample_size": 1000, "Ne": 1, "ro": 8e-6, "mu": 8e-4, "Tau": 1, "Kappa": 1, "length": int(1e7)}
    out_csv, out_fig = "csv_dir", "fig_dir"
    # plot_time_error()
    # plot_closest_error()
    # scatter_panel(out_csv)
    # closest_panel(out_csv)
    plot_time_error()
    plot_closest_error()
    nb_event()
    # for mu_ro in [1, 10, 100]:
    #     params.update({"mu": mu_ro * 8e-6})
    #     method_comparaison(params, 1, mu_ro, out_csv)
        # scatter_plot(pd.read_csv(f"{out_csv}/result_{mu_ro}"), mu_ro, out_fig)
        # box_plot_time(pd.read_csv(f"{out_csv}/closest_{mu_ro}"), mu_ro, out_fig)


if __name__ == "__main__":
    main()
