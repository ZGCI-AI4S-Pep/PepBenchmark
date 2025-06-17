import numpy as np
import pandas as pd

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G',
               'H', 'I', 'K', 'L', 'M', 'N',
               'P', 'Q', 'R', 'S', 'T', 'V',
               'W', 'Y']


def combine(inputs, labels, new_inputs, new_labels):

    inputs.extend(new_inputs)
    labels = pd.concat([labels, pd.Series(new_labels, name='label')], ignore_index=True)

    return inputs, labels


def random_replace(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(len(inputs)):
        ip = inputs[idx]
        ip_list = list(ip)
        label = labels[idx]
        unpadded_len = len(ip)
        num_to_replace = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_replace, replace=False)
        replacements = np.random.choice(amino_acids, num_to_replace, replace=True)
        for i, r in zip(indices, replacements):
            ip_list[i] = r

        ip = ''.join(ip_list)

        new_inputs.append(ip)
        new_labels.append(label)

    return new_inputs, new_labels


def random_delete(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(len(inputs)):
        ip = inputs[idx]
        label = labels[idx]
        unpadded_len = len(ip)
        ip = list(ip[:unpadded_len])
        num_to_delete = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_delete, replace=False)
        for i in reversed(sorted(indices)):
            ip.pop(i)


        new_inputs.append(''.join(ip))
        new_labels.append(label)

    return new_inputs, new_labels


def random_replace_with_A(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(len(inputs)):
        ip = inputs[idx]
        ip = list(ip[:unpadded_len])
        label = labels[idx]
        unpadded_len = len(ip)
        num_to_replace = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_replace, replace=False)
        ip[indices] = 'A'

        new_inputs.append(''.join(ip))
        new_labels.append(label)

    return new_inputs, new_labels


def random_swap(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(len(inputs)):
        ip = inputs[idx]
        label = labels[idx]
        unpadded_len = len(ip)
        ip = list(ip)
        num_to_swap = round(unpadded_len * factor)
        indices = np.random.choice(range(1, unpadded_len, 2), num_to_swap, replace=False)
        for i in indices:
            ip[i-1], ip[i] = ip[i], ip[i-1]


        new_inputs.append(''.join(ip))
        new_labels.append(label)

    return new_inputs, new_labels


def random_insertion_with_A(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(len(inputs)):
        ip = inputs[idx]
        label = labels[idx]
        unpadded_len = len(ip)
        ip = list(ip)
        num_to_insert = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_insert, replace=False)
        for i in indices:
            ip.insert(i, 'A')

        new_inputs.append(''.join(ip))
        new_labels.append(label)

    return new_inputs, new_labels




