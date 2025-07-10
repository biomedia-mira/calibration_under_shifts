# From robustness repository. The original one is incompatible with current version of torchvision.

import sys, os
import json
import numpy as np
import pandas as pd
import urllib
from collections import OrderedDict, Counter
import operator
import networkx as nx
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from PIL import Image

REQUIRED_FILES = ["dataset_class_info.json", "class_hierarchy.txt", "node_names.txt"]
BREEDS_URL = "https://github.com/MadryLab/BREEDS-Benchmarks/blob/master/imagenet_class_hierarchy/modified/"


def setup_breeds(info_dir, url=BREEDS_URL):
    print(f"Downloading files from {url} to {info_dir}")
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    for f in REQUIRED_FILES:
        urllib.request.urlretrieve(f"{url}/{f}?raw=true", os.path.join(info_dir, f))


class ClassHierarchy:
    """
    Class representing a general ImageNet-style hierarchy.
    """

    def __init__(self, info_dir, root_wnid="n00001740"):
        """
        Args:
            info_dir (str) : Path to hierarchy information files. Contains a
                "class_hierarchy.txt" file with one edge per line, a
                "node_names.txt" mapping nodes to names, and "class_info.json".
        """

        for f in REQUIRED_FILES:
            if not os.path.exists(os.path.join(info_dir, f)):
                raise Exception(
                    f"Missing files: {info_dir} does not contain required file {f}"
                )

        # Details about dataset class names (leaves), IDS
        with open(os.path.join(info_dir, "dataset_class_info.json")) as f:
            class_info = json.load(f)

        # Hierarchy represented as edges between parent & child nodes.
        with open(os.path.join(info_dir, "class_hierarchy.txt")) as f:
            edges = [l.strip().split() for l in f.readlines()]

        # Information (names, IDs) for intermediate nodes in hierarchy.
        with open(os.path.join(info_dir, "node_names.txt")) as f:
            mapping = [l.strip().split("\t") for l in f.readlines()]

        # Original dataset classes
        self.LEAF_IDS = [c[1] for c in class_info]
        self.LEAF_ID_TO_NAME = {c[1]: c[2] for c in class_info}
        self.LEAF_ID_TO_NUM = {c[1]: c[0] for c in class_info}
        self.LEAF_NUM_TO_NAME = {c[0]: c[2] for c in class_info}

        # Full hierarchy
        self.HIER_NODE_NAME = {w[0]: w[1] for w in mapping}
        self.graph = self._make_parent_graph(self.LEAF_IDS, edges)

        # Arrange nodes in levels (top-down)
        self.node_to_level = self._make_level_dict(self.graph, root=root_wnid)
        self.level_to_nodes = {}
        for k, v in self.node_to_level.items():
            if v not in self.level_to_nodes:
                self.level_to_nodes[v] = []
            self.level_to_nodes[v].append(k)

    @staticmethod
    def _make_parent_graph(nodes, edges):
        """
        Obtain networkx representation of class hierarchy.

        Args:
            nodes [str] : List of node names to traverse upwards.
            edges [(str, str)] : Tuples of parent-child pairs.

        Return:
            networkx representation of the graph.
        """

        # create full graph
        full_graph_dir = {}
        for p, c in edges:
            if p not in full_graph_dir:
                full_graph_dir[p] = {c: 1}
            else:
                full_graph_dir[p].update({c: 1})

        FG = nx.DiGraph(full_graph_dir)

        # perform backward BFS to get the relevant graph
        graph_dir = {}
        todo = [n for n in nodes if n in FG.nodes()]  # skip nodes not in graph
        while todo:
            curr = todo
            todo = []
            for w in curr:
                for p in FG.predecessors(w):
                    if p not in graph_dir:
                        graph_dir[p] = {w: 1}
                    else:
                        graph_dir[p].update({w: 1})
                    todo.append(p)
            todo = set(todo)

        return nx.DiGraph(graph_dir)

    @staticmethod
    def _make_level_dict(graph, root):
        """
        Map nodes to their level within the hierarchy (top-down).

        Args:
            graph (networkx graph( : Graph representation of the hierarchy
            root (str) : Hierarchy root.

        Return:
            Dictionary mapping node names to integer level.
        """

        level_dict = {}
        todo = [(root, 0)]  # (node, depth)
        while todo:
            curr = todo
            todo = []
            for n, d in curr:
                if n not in level_dict:
                    level_dict[n] = d
                else:
                    level_dict[n] = max(d, level_dict[n])  # keep longest path
                for c in graph.successors(n):
                    todo.append((c, d + 1))

        return level_dict

    def leaves_reachable(self, n):
        """
        Determine the leaves (ImageNet classes) reachable for a give node.

        Args:
            n (str) : WordNet ID of node

        Returns:
            leaves (list): List of WordNet IDs of the ImageNet descendants
        """
        leaves = set()
        todo = [n]
        while todo:
            curr = todo
            todo = []
            for w in curr:
                for c in self.graph.successors(w):
                    if c in self.LEAF_IDS:
                        leaves.add(c)
                    else:
                        todo.append(c)
            todo = set(todo)

        # If the node itself is an ImageNet node
        if n in self.LEAF_IDS:
            leaves = leaves.union([n])
        return leaves

    def node_name(self, n):
        """
        Determine the name of a node.
        """
        if n in self.HIER_NODE_NAME:
            return self.HIER_NODE_NAME[n]
        else:
            return n

    def print_node_info(self, nodes):
        """
        Prints basic information (name, number of ImageNet descendants)
        about a given set of nodes.

        Args:
            nodes (list) : List of WordNet IDs for relevant nodes
        """

        for n in nodes:
            if n in self.HIER_NODE_NAME:
                print_str = f"{n}: {self.HIER_NODE_NAME[n]}"
            else:
                print_str = n

            print_str += f" ({len(self.leaves_reachable(n))})"
            print(print_str)

    def traverse(self, nodes, direction="down", depth=100):
        """
        Find all nodes accessible from a set of given nodes.

        Args:
            nodes (list) : List of WordNet IDs for relevant nodes
            direction ("up"/"down"): Traversal direction
            depth (int): Maximum depth to traverse (from nodes)

        Returns:
            Set of nodes reachable within the desired depth, in the
            desired direction.
        """

        if not nodes or depth == 0:
            return nodes

        todo = []
        for n in nodes:
            if direction == "down":
                todo.extend(self.graph.successors(n))
            else:
                todo.extend(self.graph.predecessors(n))
        return nodes + self.traverse(todo, direction=direction, depth=depth - 1)

    def get_nodes_at_level(self, L, ancestor=None):
        """
        Find all superclasses at a specified depth within a subtree
        of the hierarchy.

        Args:
            L (int): Depth in hierarchy (from root node)
            ancestor (str): (optional) WordNet ID that can be used to
                            restrict the subtree in which superclasses
                            are found

        Returns:
            nodes (list): Set of superclasses at that depth in
                                   the hierarchy
        """
        if ancestor is not None:
            valid = set(self.traverse([ancestor], direction="down"))

        nodes = set(
            [v for v in self.level_to_nodes[L] if ancestor is None or v in valid]
        )
        return nodes


class BreedsDatasetGenerator:
    """
    Class for generating datasets from ImageNet superclasses.
    """

    def __init__(self, info_dir, root_wnid="n00001740"):
        self.hierarchy = ClassHierarchy(info_dir, root_wnid=root_wnid)

    def split_superclass(
        self,
        superclass_wnid,
        Nsubclasses,
        balanced,
        split_type,
        rng=np.random.RandomState(2),
    ):
        """
        Split superclass into two disjoint sets of subclasses.

        Args:
            superclass_wnid (str): WordNet ID of superclass node
            Nsubclasses (int): Number of subclasses per superclass
                               (not used when balanced is True)
            balanced (bool): Whether or not the dataset should be
                             balanced over superclasses
            split_type ("good"/"bad"/"rand"): Whether the subclass
                             partitioning should be more or less
                             adversarial or random
            rng (RandomState): Random number generator

        Returns:
            class_ranges (tuple): Tuple of lists of subclasses
        """

        # Find a descendant of the superclass that has at least two children
        G = self.hierarchy.graph
        node, desc = superclass_wnid, sorted(list(G.successors(superclass_wnid)))
        while len(desc) == 1:
            node = desc[0]
            desc = sorted(list(G.successors(node)))

        # Map each descendant to its ImageNet subclasses
        desc_map = {}
        for d in desc:
            desc_map[d] = sorted(list(self.hierarchy.leaves_reachable(d)))

        # Map sorted by nodes that have the maximum number of children
        desc_sorted = sorted(desc_map.items(), key=lambda x: -len(x[1]))

        # If not balanced, we will pick as many subclasses as possible
        # from this superclass (ignoring Nsubclasses)
        assert Nsubclasses >= 2
        if not balanced:
            S = sum([len(d) for d in desc_map.values()])
            assert S >= Nsubclasses
            Nsubclasses = S
            if Nsubclasses % 2 != 0:
                Nsubclasses -= 1

        # Split superclasses into two disjoint sets
        assert Nsubclasses % 2 == 0
        Nh = Nsubclasses // 2

        if split_type == "rand":
            # If split is random, aggregate all subclasses, subsample
            # the desired number, and then partition into two
            desc_node_list = []
            for d in desc_map.values():
                desc_node_list.extend(d)
            sel = rng.choice(sorted(desc_node_list), Nh * 2, replace=False)
            split = (sel[:Nh], sel[Nh:])
        else:
            # If split is good, we will partition similar subclasses across
            # both domains. If it is bad, similar subclasses will feature in
            # one or the other

            split, spare = ([], []), []

            for k, v in desc_sorted:
                l = [len(s) for s in split]
                if split_type == "bad":
                    if l[0] <= l[1] and l[0] < Nh:
                        if len(v) > Nh - l[0]:
                            spare.extend(v[Nh - l[0] :])
                        split[0].extend(v[: Nh - l[0]])
                    elif l[1] < Nh:
                        if len(v) > Nh - l[1]:
                            spare.extend(v[Nh - l[1] :])
                        split[1].extend(v[: Nh - l[1]])
                else:
                    if len(v) == 1:
                        i1 = 1 if l[0] < Nh else 0
                    else:
                        i1 = min(len(v) // 2, Nh - l[0])

                    if l[0] < Nh:
                        split[0].extend(v[:i1])
                    if l[1] < Nh:
                        split[1].extend(v[i1 : i1 + Nh - l[1]])

            if split_type == "bad":
                l = [len(s) + 0 for s in split]
                assert max(l) == Nh
                if l[0] < Nh:
                    split[0].extend(spare[: Nh - l[0]])
                if l[1] < Nh:
                    split[1].extend(spare[: Nh - l[1]])

        assert len(split[0]) == len(split[1]) and not set(split[0]).intersection(
            split[1]
        )
        class_ranges = (
            [self.hierarchy.LEAF_ID_TO_NUM[s] for s in split[0]],
            [self.hierarchy.LEAF_ID_TO_NUM[s] for s in split[1]],
        )

        return class_ranges

    def get_superclasses(
        self,
        level,
        Nsubclasses=None,
        split=None,
        ancestor=None,
        balanced=True,
        random_seed=2,
        verbose=False,
    ):
        """
        Obtain a dataset composed of ImageNet superclasses with a desired
        set of properties.
        (Optional) By specifying a split, one can parition the subclasses
        into two disjoint datasets (with the same superclasses).

        Args:
            level (int): Depth in hierarchy (from root node)
            Nsubclasses (int): Minimum number of subclasses per superclass
            balanced (bool): Whether or not the dataset should be
                             balanced over superclasses
            split ("good"/"bad"/"rand"/None): If None, superclasses are
                             not partitioned into two disjoint datasets.
                             If not None, determines whether the subclass
                             partitioning should be more or less
                             adversarial or random
            rng (RandomState): Random number generator

        Returns:
            superclasses (list): WordNet IDs of superclasses
            subclass_splits (tuple): Tuple entries correspond to the source
                                     and target domains respectively. A
                                     tuple entry is a list, where each
                                     element is a list of subclasses to
                                     be included in a given superclass in
                                     that domain. If split is None,
                                     the second tuple element is empty.
            label_map (dict): Map from (super)class number to superclass name
        """

        rng = np.random.RandomState(random_seed)
        hierarchy = self.hierarchy

        # Identify superclasses at this level
        rel_nodes = sorted(list(hierarchy.get_nodes_at_level(level, ancestor=ancestor)))
        if verbose:
            hierarchy.print_node_info(rel_nodes)

        # Count number of subclasses
        in_desc = []
        for n in rel_nodes:
            dcurr = self.hierarchy.leaves_reachable(n)
            in_desc.append(sorted(list(dcurr)))
        min_desc = np.min([len(d) for d in in_desc])
        assert min_desc > 0

        # Determine number of subclasses to include per superclass
        if Nsubclasses is None:
            if balanced:
                Nsubclasses = min_desc
                if Nsubclasses % 2 != 0:
                    Nsubclasses = max(2, Nsubclasses - 1)
            else:
                Nsubclasses = 1 if split is None else 2

        # Find superclasses that have sufficient subclasses
        superclass_idx = [
            i for i in range(len(rel_nodes)) if len(in_desc[i]) >= Nsubclasses
        ]
        superclasses, all_subclasses = [rel_nodes[i] for i in superclass_idx], [
            in_desc[i] for i in superclass_idx
        ]

        # Superclass names
        label_map = {}
        for ri, r in enumerate(superclasses):
            label_map[ri] = self.hierarchy.node_name(r)

        if split is None:

            if balanced:
                Ns = [Nsubclasses] * len(all_subclasses)
            else:
                Ns = [len(d) for d in all_subclasses]
            wnids = [
                list(rng.choice(d, n, replace=False))
                for n, d in zip(Ns, all_subclasses)
            ]
            subclass_ranges = [
                [self.hierarchy.LEAF_ID_TO_NUM[w] for w in c] for c in wnids
            ]
            subclass_splits = (subclass_ranges, [])
        else:
            subclass_splits = ([], [])
            for sci, sc in enumerate(sorted(superclasses)):
                class_tup = self.split_superclass(
                    sc,
                    Nsubclasses=Nsubclasses,
                    balanced=balanced,
                    split_type=split,
                    rng=rng,
                )
                subclass_splits[0].append(class_tup[0])
                subclass_splits[1].append(class_tup[1])

        return superclasses, subclass_splits, label_map


def print_dataset_info(superclasses, subclass_splits, label_map, label_map_sub):
    """
    Obtain a dataframe with information about the
    superclasses/subclasses included in the dataset.

    Args:
    superclasses (list): WordNet IDs of superclasses
    subclass_splits (tuple): Tuple entries correspond to the source
                             and target domains respectively. A
                             tuple entry is a list, where each
                             element is a list of subclasses to
                             be included in a given superclass in
                             that domain. If split is None,
                             the second tuple element is empty.
    label_map (dict): Map from (super)class number to superclass name
    label_map_sub (dict):  Map from subclass number to subclass name
                              (equivalent to label map for original dataset)
    Returns:
        dataDf (pandas DataFrame): Columns contain relevant information
                                about the datast

    """

    def print_names(class_idx):
        return [f'{label_map_sub[r].split(",")[0]} ({r})' for r in class_idx]

    data = {"superclass": []}
    contains_split = len(subclass_splits[1])
    if contains_split:
        data.update({"subclasses (source)": [], "subclasses (target)": []})
    else:
        data.update({"subclasses": []})

    for i, (k, v) in enumerate(label_map.items()):
        data["superclass"].append(f"{v}")
        if contains_split:
            data["subclasses (source)"].append(print_names(subclass_splits[0][i]))
            data["subclasses (target)"].append(print_names(subclass_splits[1][i]))
        else:
            data["subclasses"].append(print_names(subclass_splits[0][i]))

    dataDf = pd.DataFrame(data)
    return dataDf


# Some standard datasets from the BREEDS paper.
def make_entity13(info_dir, split=None):
    """
    Obtain superclass/subclass information for the ENTITY-13 dataset
    Args:
        info_dir (str) : Path to ImageNet information files
        split ("good"/"bad"/"rand"/None): Nature of subclass
    Returns:
        superclasses (list): WordNet IDs of superclasses
        subclass_splits (tuple): Tuple entries correspond to the source
                                 and target domains respectively. A
                                 tuple entry is a list, where each
                                 element is a list of subclasses to
                                 be included in a given superclass in
                                 that domain. If split is None,
                                 the second tuple element is empty.
        label_map (dict): Map from (super)class number to superclass name

    """

    DG = BreedsDatasetGenerator(info_dir)
    return DG.get_superclasses(
        level=3,
        ancestor=None,
        Nsubclasses=20,
        split=split,
        balanced=True,
        random_seed=2,
        verbose=False,
    )


def make_entity30(info_dir, split=None):
    """
    Obtain superclass/subclass information for the ENTITY-30 dataset
    Args:
        info_dir (str) : Path to ImageNet information files
        split ("good"/"bad"/"rand"/None): Nature of subclass
    Returns:
        superclasses (list): WordNet IDs of superclasses
        subclass_splits (tuple): Tuple entries correspond to the source
                                 and target domains respectively. A
                                 tuple entry is a list, where each
                                 element is a list of subclasses to
                                 be included in a given superclass in
                                 that domain. If split is None,
                                 the second tuple element is empty.
        label_map (dict): Map from (super)class number to superclass name

    """
    DG = BreedsDatasetGenerator(info_dir)
    return DG.get_superclasses(
        level=4,
        ancestor=None,
        Nsubclasses=8,
        split=split,
        balanced=True,
        random_seed=2,
        verbose=False,
    )


def make_living17(info_dir, split=None):
    """
    Obtain superclass/subclass information for the LIVING-17 dataset
    Args:
        info_dir (str) : Path to ImageNet information files
        split ("good"/"bad"/"rand"/None): Nature of subclass
    Returns:
        superclasses (list): WordNet IDs of superclasses
        subclass_splits (tuple): Tuple entries correspond to the source
                                 and target domains respectively. A
                                 tuple entry is a list, where each
                                 element is a list of subclasses to
                                 be included in a given superclass in
                                 that domain. If split is None,
                                 the second tuple element is empty.
        label_map (dict): Map from (super)class number to superclass name

    """
    DG = BreedsDatasetGenerator(info_dir)
    return DG.get_superclasses(
        level=5,
        ancestor="n00004258",
        Nsubclasses=4,
        split=split,
        balanced=True,
        random_seed=2,
        verbose=False,
    )


def make_nonliving26(info_dir, split=None):
    """
    Obtain superclass/subclass information for the NONLIVING-26 dataset.
    Args:
        info_dir (str) : Path to ImageNet information files
        split ("good"/"bad"/"rand"/None): Nature of subclass
    Returns:
        superclasses (list): WordNet IDs of superclasses
        subclass_splits (tuple): Tuple entries correspond to the source
                                 and target domains respectively. A
                                 tuple entry is a list, where each
                                 element is a list of subclasses to
                                 be included in a given superclass in
                                 that domain. If split is None,
                                 the second tuple element is empty.
        label_map (dict): Map from (super)class number to superclass name

    """
    DG = BreedsDatasetGenerator(info_dir)
    return DG.get_superclasses(
        level=5,
        ancestor="n00021939",
        Nsubclasses=4,
        split=split,
        balanced=True,
        random_seed=2,
        verbose=False,
    )


def make_loaders(
    workers,
    batch_size,
    transforms,
    data_path,
    data_aug=True,
    custom_class=None,
    dataset="",
    label_mapping=None,
    subset=None,
    subset_type="rand",
    subset_start=0,
    val_batch_size=None,
    only_val=False,
    shuffle_train=True,
    shuffle_val=True,
    seed=1,
    custom_class_args=None,
):
    """
    **INTERNAL FUNCTION**

    This is an internal function that makes a loader for any dataset. You
    probably want to call dataset.make_loaders for a specific dataset,
    which only requires workers and batch_size. For example:

    >>> cifar_dataset = CIFAR10('/path/to/cifar')
    >>> train_loader, val_loader = cifar_dataset.make_loaders(workers=10, batch_size=128)
    >>> # train_loader and val_loader are just PyTorch dataloaders
    """
    print(f"==> Preparing dataset {dataset}..")
    transform_train, transform_test = transforms
    if not data_aug:
        transform_train = transform_test

    if not val_batch_size:
        val_batch_size = batch_size

    if not custom_class:
        train_path = os.path.join(data_path, "train")
        test_path = os.path.join(data_path, "val")
        if not os.path.exists(test_path):
            test_path = os.path.join(data_path, "test")

        if not os.path.exists(test_path):
            raise ValueError(
                "Test data must be stored in dataset/test or {0}".format(test_path)
            )

        if not only_val:
            train_set = ImageFolder(
                root=train_path, transform=transform_train, label_mapping=label_mapping
            )
        test_set = ImageFolder(
            root=test_path, transform=transform_test, label_mapping=label_mapping
        )
    else:
        if custom_class_args is None:
            custom_class_args = {}
        if not only_val:
            train_set = custom_class(
                root=data_path,
                train=True,
                download=True,
                transform=transform_train,
                **custom_class_args,
            )
        test_set = custom_class(
            root=data_path,
            train=False,
            download=True,
            transform=transform_test,
            **custom_class_args,
        )

    if not only_val:
        attrs = ["samples", "train_data", "data"]
        vals = {attr: hasattr(train_set, attr) for attr in attrs}
        assert any(vals.values()), f"dataset must expose one of {attrs}"
        train_sample_count = len(getattr(train_set, [k for k in vals if vals[k]][0]))

    if (not only_val) and (subset is not None) and (subset <= train_sample_count):
        assert not only_val
        if subset_type == "rand":
            rng = np.random.RandomState(seed)
            subset = rng.choice(
                list(range(train_sample_count)),
                size=subset + subset_start,
                replace=False,
            )
            subset = subset[subset_start:]
        elif subset_type == "first":
            subset = np.arange(subset_start, subset_start + subset)
        else:
            subset = np.arange(train_sample_count - subset, train_sample_count)

        train_set = Subset(train_set, subset)

    if not only_val:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=workers,
            pin_memory=True,
        )

    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=shuffle_val,
        num_workers=workers,
        pin_memory=True,
    )

    if only_val:
        return None, test_loader

    return train_loader, test_loader


class DataSet(object):
    """
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function.
    """

    def __init__(self, ds_name, data_path, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : path to the dataset
            num_classes (int) : *required kwarg*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg*, the mean to normalize the
                dataset with (e.g.  :samp:`ch.tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg*, the standard deviation to
                normalize the dataset with (e.g. :samp:`ch.tensor([0.2023,
                0.1994, 0.2010])` for CIFAR-10)
            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*,
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        """
        required_args = [
            "num_classes",
            "mean",
            "std",
            "transform_train",
            "transform_test",
        ]
        optional_args = ["custom_class", "label_mapping", "custom_class_args"]

        missing_args = set(required_args) - set(kwargs.keys())
        if len(missing_args) > 0:
            raise ValueError("Missing required args %s" % missing_args)

        extra_args = set(kwargs.keys()) - set(required_args + optional_args)
        if len(extra_args) > 0:
            raise ValueError("Got unrecognized args %s" % extra_args)
        final_kwargs = {k: kwargs.get(k, None) for k in required_args + optional_args}

        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(final_kwargs)

    def override_args(self, default_args, kwargs):
        """
        Convenience method for overriding arguments. (Internal)
        """
        for k in kwargs:
            if not (k in default_args):
                continue
            req_type = type(default_args[k])
            no_nones = (default_args[k] is not None) and (kwargs[k] is not None)
            if no_nones and (not isinstance(kwargs[k], req_type)):
                raise ValueError(f"Argument {k} should have type {req_type}")
        return {**default_args, **kwargs}

    def make_loaders(
        self,
        workers,
        batch_size,
        data_aug=True,
        subset=None,
        subset_start=0,
        subset_type="rand",
        val_batch_size=None,
        only_val=False,
        shuffle_train=True,
        shuffle_val=True,
        subset_seed=None,
    ):
        """
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            seed (int) : only used if `subset == "rand"`; allows one to fix
                the random seed used to generate the subset (defaults to 1).
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader
            shuffle_train (bool) : Whether or not to shuffle the training data
                in the returned DataLoader.
            shuffle_val (bool) : Whether or not to shuffle the test data in the
                returned DataLoader.

        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:

            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128)
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        """
        transforms = (self.transform_train, self.transform_test)
        return make_loaders(
            workers=workers,
            batch_size=batch_size,
            transforms=transforms,
            data_path=self.data_path,
            data_aug=data_aug,
            dataset=self.ds_name,
            label_mapping=self.label_mapping,
            custom_class=self.custom_class,
            val_batch_size=val_batch_size,
            subset=subset,
            subset_start=subset_start,
            subset_type=subset_type,
            only_val=only_val,
            seed=subset_seed,
            shuffle_train=shuffle_train,
            shuffle_val=shuffle_val,
            custom_class_args=self.custom_class_args,
        )


def custom_label_mapping(classes, class_to_idx, ranges):

    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(ranges):
            if idx in range_set:
                mapping[class_name] = new_idx

    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping


def get_label_mapping(ranges):
    def label_mapping(classes, class_to_idx):
        return custom_label_mapping(classes, class_to_idx, ranges=ranges)

    return label_mapping


class CustomImageNet(DataSet):
    """
    CustomImagenet Dataset

    A subset of ImageNet with the user-specified labels

    To initialize, just provide the path to the full ImageNet dataset
    along with a list of lists of wnids to be grouped together
    (no special formatting required).

    """

    def __init__(self, data_path, custom_grouping, **kwargs):
        """ """
        ds_name = "custom_imagenet"
        ds_kwargs = {
            "num_classes": len(custom_grouping),
            "mean": torch.tensor([0.4717, 0.4499, 0.3837]),
            "std": torch.tensor([0.2600, 0.2516, 0.2575]),
            "custom_class": None,
            "label_mapping": get_label_mapping(custom_grouping),
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(CustomImageNet, self).__init__(ds_name, data_path, **ds_kwargs)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root,
        loader,
        extensions,
        transform=None,
        target_transform=None,
        label_mapping=None,
    ):
        classes, class_to_idx = self._find_classes(root)
        if label_mapping is not None:
            classes, class_to_idx = label_mapping(classes, class_to_idx)

        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + root + "\n"
                    "Supported extensions are: " + ",".join(extensions)
                )
            )

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=pil_loader,
        label_mapping=None,
    ):
        super(ImageFolder, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
            label_mapping=label_mapping,
        )
        self.imgs = self.samples
