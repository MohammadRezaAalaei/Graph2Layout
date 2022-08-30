import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils import process_data, process_graph, show_tensor_images, seperate_graphs, process
DEVICE='cpu'

def generate(t_path, adj_path, b_path, ent_path, centers_path):
    """
    This function generates architectural space layouts given input predefined features
    :param t_path: File path of nodes' type features
    :param adj_path: File path of adjacency matrix
    :param b_path: File path of floorplan boundary
    :param ent_path: File path of floorplan's entrace location
    :param centers_path: File path of nodes' location features
    :return: Space layout design, different colors specified for each space
    """


    t_features, coo, boundary, entrance, centers, index = process_data(
        t_path,
        adj_path,
        b_path,
        ent_path,
        centers_path
    )

    data_t = [process_graph('data_t', t_features)]
    indices = [process_graph('data_idx', index)]
    data_b = [process_graph('data_b', boundary)]
    data_m = [process_graph('data_m', centers)]
    ix = indices[0].x
    idx_data = []
    for i in ix.unique():
        idx_data.append(len(ix[ix == i]))
    idx_data = torch.tensor(idx_data).unsqueeze(0)
    print(idx_data)




    def show_plot(fake, condition, centers, types, indices, ix, cur_batch_size):
        """
        Plots generated space layouts given condition, space locations
        and types, graph indices, node indices, and current batch size
        :param fake:
        :param condition:
        :param centers:
        :param types:
        :param indices:
        :param ix:
        :param cur_batch_size:
        :return: colors
        """

        # remove overlaps
        cond_list = seperate_graphs(ix, condition)
        torch.set_printoptions(threshold=np.inf)
        typ = torch.where(types > 0.5)[-1].view(fake.shape[0], 1, 1, 1) + 1
        idx = typ
        fake = (idx * fake)
        centers = (idx * centers)

        center_list = seperate_graphs(ix, centers)
        fake_list = seperate_graphs(ix, fake)
        typ_list = seperate_graphs(ix, typ)
        condition = [c[0] for c in cond_list]
        fake_temp = []
        center_temp = []
        color = []
        for c, i, t in zip(center_list, fake_list, typ_list):
            space_i = i
            spaces_i = []
            centers = []
            for k in range(len(space_i)):


                rgb, color_rgb = process(space_i[k], t[k])

                spaces_i.append(rgb)

                # center______________________________________

                center, color_rgb = process(c[k], t[k])
                color.append(color_rgb)
                centers.append(center)

            fake_temp.append(torch.sum(torch.stack(spaces_i), dim=0))
            center_temp.append(torch.sum(torch.stack(centers), dim=0))

        condition = torch.stack(condition).unsqueeze(1)
        fake = torch.stack(fake_temp)
        centers = torch.stack(center_temp)
        plt.figure(figsize=[10, 10])
        show_tensor_images(condition.to('cpu').detach())
        plt.title("Footprint condition")

        plt.figure(figsize=[10, 10])
        show_tensor_images(condition.to('cpu').detach())
        show_tensor_images(centers.to('cpu').detach(), alpha=0.5)
        plt.title("Topological condition")

        plt.figure(figsize=[10, 10])
        show_tensor_images(fake.to('cpu'))
        plt.title("fake")
        plt.show()
        return color

    for condition, m, t, idx, ix in zip(data_b, data_m, data_t, indices, idx_data):
        print(t)
        t = t.to(DEVICE)
        cur_batch_size = len(t.x)

        condition = condition.to(DEVICE)
        boundary = condition.x.unsqueeze(1)
        m = m.to(DEVICE)
        centers = m.x.unsqueeze(1)

        node_t = t.x.type(torch.float)
        node_t = node_t.view(node_t.shape[0], node_t.shape[1], 1, 1)
        ones = torch.ones(size=[node_t.shape[0], node_t.shape[1], 64, 64], dtype=torch.float,
                          device=torch.device(DEVICE))
        node_t = node_t * ones
        print(condition.x.shape)
        condition0 = torch.cat([condition.x.view(condition.x.shape[1], 1, 64, 64), centers, node_t], dim=1)
        with torch.no_grad():
            print(t.edge_index.shape)
            layout = gen(condition0, t.edge_index.type(torch.long))
            threshold = 0.6
            layout[layout < threshold] = 0
            layout[layout > threshold] = 1

            color = show_plot(layout, boundary, centers, t.x, idx.x, ix, cur_batch_size)
    return layout, color


space_layout, color = generate(
    'input_types.csv',
    'input_adj.csv',
    'input_boundary.csv',
    'input_entrance.csv',
    'centers.csv'
)
layout = torch.flatten(space_layout, start_dim=1)
color_rgb = np.array(color)
color_rgb = pd.DataFrame(color_rgb)
color_rgb.to_csv('colors.csv')

layout = layout.numpy()
df = pd.DataFrame(layout)
df.to_csv('output.csv')
space_layout = space_layout.squeeze()
pt_x = []
pt_y = []

for i in space_layout:
    y, x = torch.where(i == 1)
    y = 64 - y
    pt_x.append(x)
    pt_y.append(y)

length = [len(i) for i in pt_x]
length = pd.DataFrame(length)
length.to_csv('length.csv')
space_layout = space_layout.squeeze()
pt_x = []
pt_y = []

for i in space_layout:
    y, x = torch.where(i == 1)
    y = 64 - y
    pt_x += x
    pt_y += y

pt_x = torch.stack(pt_x)
pt_y = torch.stack(pt_y)
pt_x = pt_x.numpy()
pt_y = pt_y.numpy()
pt_x = pd.DataFrame(pt_x)
pt_y = pd.DataFrame(pt_y)
pt_x.to_csv('pt_x.csv')
pt_y.to_csv('pt_y.csv')