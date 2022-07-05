"""
classe les points de recombinaisons entre discrets, detectactbles par les tests
orientés et non orientés, incompatibles, uniquement détectable par le test
orienté (4 gamètes), et non détectables
"""

import copy as cpy


class _Tree:
    """
    implémentation d'un arbre
    attributs:
        tree, dictionnaire de liste avec les noeud parents comme clé et la liste des noeuds fils comme valeurs
    """

    def __init__(self, tree):
        self.tree = cpy.deepcopy(tree)

    def add_edge(self, parent, child):
        """
        ajoute une arrêtes orienté à lattribut tree
        entrées:
            parent, int: noeuds parents
            child, int, noeud fils
        """
        if parent in self.tree:
            self.tree[parent].append(child)
        else:
            self.tree[parent] = [child]

    def equal(self, tree):
        """
        verifie l'égualité entre deux arbre
        """
        return self.tree == tree.tree


class _Unrooted(_Tree):
    """
    implémentaion d'un arbre non raciné et non orienté
    attributs:
        centers, liste d'entiers: liste centres de l'arbre, il ne peut y en avoir
        qu'un ou deux
        rooted_trees, liste d'arbre racinés: les arbres dont les
        racines sont les centres, un arbre par centre
    """

    def __init__(self, rooted):
        """
        constructeur
        entrée:
            rooted, un arbre enraciné
        """
        parent, child = rooted.tree[rooted.root]
        del rooted.tree[rooted.root]  # supression de la racine de l'arbre enraciné
        rooted.add_edge(parent, child)  # reconexion des deux noeuds fils de la racines suprimé
        super().__init__(cpy.deepcopy(rooted.tree))
        for child in rooted.tree:
            for parent in rooted.tree[child]:
                self.add_edge(parent, child)  # pour toutes les arr^tes de l'arbres l'arrêtes symétrique est ajoutées
        self.centers = self.__centers()  # initilaisation des centres
        self.rooted_trees = self.__to_rooted(self.centers)  # initialisation des arebres racinés à partir des centres

    def __leaves(self):
        """
        sorties:
            degrees, dictionnaire: les clés sont les noeuds et les veleurs leur
            degré
            leaves, liste d'entiers: liste des feuilles de l'arbres
        """
        degrees = {node: len(self.tree[node]) for node in self.tree}
        leaves = [node for node in degrees if degrees[node] == 1]
        return leaves, degrees

    def __centers(self):
        """
        sortie:
            la liste des centres de l'arbre non enraciné
        """
        nb_nodes = len(self.tree)
        leaves, degrees = self.__leaves()
        count = len(leaves)
        while count < nb_nodes:
            new_leaves = []
            for leave in leaves:
                for neighbor in self.tree[leave]:
                    degrees[neighbor] -= 1
                    if degrees[neighbor] == 1:
                        new_leaves.append(neighbor)
                degrees[leave] = 0
            count += len(new_leaves)
            leaves = cpy.copy(new_leaves)
        return leaves

    def __to_rooted(self, roots):
        """
        entrée:
            root, liste de noeuds: noeuds à partir du quel enraciné l'arbre
        sortie:
            rooted_trees, liste d'arbres enracinés: ces arbre enraciné sont
            orientés mais pas binaires (racine de degré trois)
        """
        rooted_trees = [_Rooted({}, root) for root in roots]
        for index, root in enumerate(roots):
            pile = [root]  # initialisation de l apile à partir de la racine
            while pile:  # tant que la pile n'est pas vide
                node = pile[-1]
                del pile[-1]  # on dépile
                for child in self.tree[node]:
                    if child not in rooted_trees[index].tree:
                        rooted_trees[index].add_edge(node, child)
                        pile.append(child)
        return rooted_trees

    def are_isomorphics(self, unrooted):
        """
        entrée:
            unrooted, un second arbre non enraciné
        sortie:
            booléen: vrai si deux arbres non enracinés sont isomorphes en
            ne tenant compte que des labels porté par les feuilles
        """
        if len(self.centers) != len(
                unrooted.centers):  # si deux arbre n'ont pas le même nombre de centres il ne sont pas isomorhes
            return False
        if len(self.centers) < len(unrooted.centers):  # l'  arbre avec le moins de centres n'est enraciné qu'une fois"
            reference_tree = self.rooted_trees[0]
            tmp = unrooted
        else:
            reference_tree = unrooted.rooted_trees[0]
            tmp = self
        for rooted in tmp.rooted_trees:  # pour tous les arbre racinés du second arbre non enraciné
            if rooted.are_isomorphics_canonical(
                    reference_tree):  # si au moins un de ces arbres est isomorphe à l'arbre de référence alors les deux arbres non enracinés corespondant sont isomorphes
                return True
        return False


class _Rooted(_Tree):
    """
    classe implémentant un arbre enraciné, orienté binaire ou non
    attibuts:
        root, int: racine de l'abre
    """

    def __init__(self, tree, root):
        """
        constructeur
        """
        super().__init__(tree)
        self.root = root

    def canonical(self, node):
        """
        parcours en profondeur
        entrée:
            node, int: noued courant de l'arbre
        sortie:
            str: l'expression canonique de l'arbre enraciné
        """
        if node not in self.tree:  # si le noeuds et une feuille, fin de récursion
            return "(" + str(node // 10) + str(
                node % 10)  # les noeuds sont codé sur des nombre à deux chiffres de 00 à 99, il est déconseillé de l'utilisé sure des arbres de plus de 100 noeuds
        return "(" + "".join(sorted([self.canonical(child) for child in self.tree[node]])) + ")"

    def are_isomorphics_canonical(self, tree_2):
        """
        peut être utilisé sur des arbre enraciné, orienté non binares
        entrée:
            tree2, un second arbre enraciné
        sortie:
            booléen: vrai si deux arbres enracinés sont isomorphes en utilisant
            leur expression canonique en ne tenant compte que des feuilles
        """
        return self.canonical(self.root) == tree_2.canonical(
            tree_2.root)  # si les deux expressions canoniques sont les mêmes , les arbres sont isomorphqiues

    def are_isomorphics(self, tree_2, root_1, root_2):
        """
        plus rapide que are_isomorphics_canonical mais à n'utiliser que sur des
        arbres enracinés, orientés et binaires
        entrée:
            tree2, un second arbre enraciné
            root_1, int: racine de l'arbre ou d'un sous arbre de tree_1
            root_2, int: racine de l'arbre ou d'un sous arbre de tree_2
        sortie:
            booléen: vrai si deux arbres enracinés bianires sont isomorphes
            en tenant compatible que des feuilles
        """
        if root_1 not in self.tree and root_2 not in tree_2.tree:  # si les deux racines sont des feuilles dans leur arbre réspectif
            if root_1 == root_2:  # si les individue porté par les deux feuilles des deux arbres sont les mêmes les feuilles sont isomorphe
                return True
            return False  # sinon elle ne le sont pas
        if root_1 not in self.tree or root_2 not in tree_2.tree:  # si le racine d'un des deux sous arbres et une feuille et l'autre non, les arbres ne sont pas isomorphes
            return False
        # comparaison de sous arbres gauches entre eux et des sous arbres droits entre eux \
        print( self.tree, tree_2.tree)
        return (self.are_isomorphics(tree_2, self.tree[root_1][0], tree_2.tree[root_2][0]) and
                self.are_isomorphics(tree_2, self.tree[root_1][1], tree_2.tree[root_2][1])) or \
               (self.are_isomorphics(tree_2, self.tree[root_1][0], tree_2.tree[root_2][1]) and
                self.are_isomorphics(tree_2, self.tree[root_1][1], tree_2.tree[root_2][0]))
        # comparaison des sous arbres droits avec les sous arbres gauches

    def brkpoint_type(self, rooted, topo_unrooted=True):
        """
        entrée:
            rooted: un deuxième arbre enraciné
        sortie:
            str: la classe du point de recombinaison délimité par les deux
            généalogies
        """
        if self.equal(rooted):  # dans le cas ou les deux arbres sont strictement identiques
            return "hidden"
        if self.are_isomorphics_canonical(rooted): # , self.root,
                                # rooted.root):  # si les deux généalogies sont isomorphes (labels des feuilles + topologie)
            return "silent"
        if topo_unrooted:
            unrooted_1, unrooted_2 = self.to_unrooted(), rooted.to_unrooted()  # sinon on s'intéresse à l'isomorphisme des arbres non enracinés
            if unrooted_1.are_isomorphics(
                    unrooted_2):  # deux généalogies ne sont pas isomorphe et que les arbres non enraciné le sont
                return "discret"
        return "incompatible"  # les généalogie ainsi que les arbres non enraciné corepondants ne sont pas isomorphes

    def to_unrooted(self):
        """
        sortie:
            _unrooted, l'arbre non enrainé corespondant à la généalogie
        """
        return _Unrooted(self.copy())

    def set_root(self, root):
        """
        modification de la racine
        """
        self.root = root

    def copy(self):
        """
        copie de l'instance courante
        """
        return _Rooted(tree=cpy.deepcopy(self.tree), root=self.root)

    def equal(self, rooted):
        """
        rooted: un deuxième arbre enraciné
        sortie:
            bool: vrai si les deux arbres sont égaux
        """
        return super().equal(rooted) and self.root == rooted.root


def find_dico(brkpoints, current_brkpoint, left, right):
    """
    renvoie la position du point de recombinaison dans la liste des points
    de recombinaisons par une recherche à une recherche dicothomique
    """
    middle = int((left + right) // 2)
    if brkpoints[middle] == current_brkpoint:
        return middle
    if current_brkpoint > brkpoints[middle]:
        return find_dico(brkpoints, current_brkpoint, middle, right)
    return find_dico(brkpoints, current_brkpoint, left, middle)


def trees_construction(edges, brkpoints):
    """
    construction des arbres racinés pour tout les bloques non recombinants de
    l'alignement
    entrées:
        edges: liste d'arrête d'un objet treesequence de la librairie tskit,
        une arrêtes possède quatres attributs : left, la position en 3' sur l'alignements
        des séquences,  right, la position en 5', parent et child les id des individues reliés
        par l'arrête
        brkpoints: liste des positions des points de recombinaisons présents sur l'alignement
    sortie:
        list_tree: liste d'objets de la classe _Rooted (arbres raciné, orienté et binaire représentant la généalogie)
    """
    list_tree = [_Rooted({}, None) for i in range(len(brkpoints) - 1)]  # initialisation des arbres
    for edge in edges:  # boucle parcourant les arrêtes du tree sequence
        left = find_dico(brkpoints, edge.left, 0,
                         len(brkpoints))  # cherche le points de recombinaison en amonts du premiers bloque non recombinants qui contient l'arrête
        right = left + 1
        while right < len(brkpoints) and edge.right >= brkpoints[right]:  # ajoute l'arrête à tout les arbres concernés
            list_tree[right - 1].add_edge(edge.parent, edge.child)
            right += 1
    return list_tree


def class_brkpoints(edges, events, unroot=False):
    """
    fonction principale du module
    entrées:
        edges: liste d'arrête d'un objet treesequence de la librairie tskit,
        une arrêtes possède quatres attributs : left, la position en 3' sur l'alignements
        des séquences,  right, la position en 5', parent et child les id des individues reliés
        par l'arrête
        brkpoints: positions des points de recombinaisons présents sur l'alignement
    """
    list_tree = trees_construction(edges,
                                   events)  # construction des abres de coalescence racinés, un arbre par blocs non recombinants
    rooted_1 = list_tree[0]  # arbre du premiers blocs
    class_bk = []
    rooted_1.set_root(max(rooted_1.tree))  # initialisation de la racine, noeuds le plus anciens
    for index, rooted_2 in enumerate(list_tree[1:]):
        rooted_2.set_root(max(rooted_2.tree))  # arbre du bloc suivant
        class_bk.append(
            (events[index + 1], rooted_1.brkpoint_type(rooted_2, unroot)))  # asignation du points entre les deux bloques
        rooted_1 = rooted_2.copy()  #
    return class_bk
