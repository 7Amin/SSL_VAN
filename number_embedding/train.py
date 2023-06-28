import torch
import random
import numpy as np
import json
from number_embedding.model import NumberEmbedding


def calculate_accuracy(model, cluster_range=600):
    model.eval()
    total_examples = 0
    correct_predictions = 0
    batch_size = cluster_range
    input_data = []
    for i in range(cluster_range):
        input_data.append(i)
    input_data = np.array(input_data)
    input_data = torch.from_numpy(input_data)
    labels = torch.nn.functional.one_hot(input_data, cluster_range).float()
    with torch.no_grad():
        output, embed = model(input_data)  # Forward pass
        _, predicted = torch.max(output, dim=1)  # Get the predicted numbers

        total_examples += batch_size
        correct_predictions += torch.sum(torch.eq(predicted, labels)).item()

    accuracy = correct_predictions / total_examples
    return accuracy


def train(batch_size=100, cluster_range=600, dim=128, epoch_number=1000):
    model = NumberEmbedding(dim=dim, output_size=cluster_range)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    last_loss = 0.
    for e in range(epoch_number):
        input_data = []
        for i in range(batch_size):
            input_data.append(random.randint(0, cluster_range - 1))
        input_data = np.array(input_data)
        input_data = torch.from_numpy(input_data)
        labels = torch.nn.functional.one_hot(input_data, cluster_range).float()
        optimizer.zero_grad()
        pred, embed = model(input_data)

        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        print('Epoch number {} loss: {}'.format(e + 1, last_loss))

    input_data = []
    for i in range(cluster_range):
        input_data.append(i)
    accuracy = calculate_accuracy(model, cluster_range=600)
    print(f"Accuracy: {accuracy}")
    input_data = np.array(input_data)
    input_data = torch.from_numpy(input_data)
    _, embed = model(input_data)
    res = dict()
    embed = embed.detach().numpy()
    for i in range(cluster_range):
        res[i] = list(embed[i])

    res = {k: [float(val) for val in v] for k, v in res.items()}

    with open('embedded_values.json', "w") as outfile:
        json.dump(res, outfile, indent=4)

    return last_loss


train(batch_size=512, cluster_range=600, dim=128, epoch_number=1000)
