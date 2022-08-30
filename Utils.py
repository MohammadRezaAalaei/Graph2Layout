
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def process_data(t_path, adj_path, b_path, ent_path, centers_path):
    """
    :param t_path:
    :param adj_path:
    :param b_path:
    :param ent_path:
    :param centers_path:
    :return: type features, coo matrix for specifying graphs' edges, plan's boundary, plan's entrance, spaces location, index
    """

    t_features = pd.read_csv(t_path)

    t_features = torch.Tensor(pd.DataFrame(t_features).to_numpy())

    adj = pd.read_csv(adj_path)

    adj = pd.DataFrame(adj).to_numpy()

    adj = torch.Tensor(adj)

    coo = (adj > 0).nonzero().t()

    boundary = pd.read_csv(b_path)

    boundary = pd.DataFrame(boundary).to_numpy()

    boundary = torch.Tensor(boundary).view(64, 64)

    entrance = pd.read_csv(ent_path)

    entrance = pd.DataFrame(entrance).to_numpy()

    entrance = torch.Tensor(entrance).view(64, 64)

    index = torch.zeros(len(adj))

    # boundary[boundary!=0] = 1.3500
    boundary[entrance != 0] = 0.3

    centers = pd.read_csv(centers_path)
    centers = pd.DataFrame(centers).to_numpy()
    centers = torch.Tensor(centers).T.view(len(adj), 64, 64)

    boundary = torch.stack([boundary.clone() for _ in range(len(adj))]).view(1, len(adj), 64, 64)

    return t_features, coo, boundary, entrance, centers, index

def process_graph(path, data):
    """
    receives raw data and converts it to meaningful graph representation
    :param path:
    :param data:
    :return: processed_data
    """
    processed_data = Data(
        x=data,
        edge_index=coo
    )
    processed_data = processed_data.pin_memory()
    return (processed_data)

def show_tensor_images(image_tensor, num_images=4, size=(1, 64, 64), alpha=1):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.

    :param image_tensor:
    :param num_images:
    :param size:
    :param alpha:
    :return: None
    """
    image_unflat = image_tensor
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze(), alpha=alpha)

def seperate_graphs(indices, input_data, show_plot=False):
    """
    Recieves flatten nodes, as well as graph indices and separates graphs based on related floorplan
    :param indices:
    :param input_data:
    :param show_plot:
    :return: split data
    """
    if not show_plot:
        indices = list(indices)
        data = torch.split(input_data.squeeze(), indices)
        return data[:]

# rgb = [dark_gr, green, blue, yellow, fosphor, orange, red, pink, brown ,yellow_green, sky_blue]
red = [0, 0, 0, 255, 0, 255, 255, 255, 153, 153, 153]
green = [102, 255, 0, 255, 255, 128, 0, 51, 76, 255, 204]
blue = [0, 0, 255, 0, 255, 0, 0, 255, 0, 51, 255]
def process(node, node_type):
    r = torch.zeros_like(node)
    g = torch.zeros_like(node)
    b = torch.zeros_like(node)
    r[node != 0] = red[node_type]
    g[node != 0] = green[node_type]
    b[node != 0] = blue[node_type]
    rr = red[node_type]
    gg = green[node_type]
    bb = blue[node_type]
    r = node * r
    g = node * g
    b = node * b
    color = [rr, gg, bb]
    rgb = torch.stack([r, g, b]).detach()
    return rgb, color