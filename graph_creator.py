import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool


class AdaptiveSingleScaleGeometricConsistencyGraphCreator:
    def __init__(self, embedding_dim=32, sparsity_sigma=1.0):
        self.embedding_dim = embedding_dim
        self.sparsity_sigma = sparsity_sigma
        self.graph_list = None
        self.p = None
        self.global_min = None
        self.global_max = None
        self.global_k = None

    def load_data(self, sensor_files, label_file):
        sensor_data = []
        min_dim = float('inf')
        max_dim = 0

        for file in sensor_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Sensor file not found: {file}")

            with open(file, 'r') as f:
                lines = f.readlines()

            data_list = []
            for line in lines:
                values = [float(x.strip()) for x in line.strip().split(',') if x.strip()]
                if values:
                    data_list.append(values)
                    dim = len(values)
                    if dim < min_dim:
                        min_dim = dim
                    if dim > max_dim:
                        max_dim = dim

            data = torch.tensor(data_list, dtype=torch.float32)
            sensor_data.append(data)

        n_samples_list = [data.shape[0] for data in sensor_data]
        if len(set(n_samples_list)) > 1:
            raise ValueError(f"Inconsistent sample counts: {n_samples_list}")

        n_samples = n_samples_list[0]

        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        with open(label_file, 'r') as f:
            label_lines = f.readlines()

        labels_list = []
        for line in label_lines:
            values = [int(x.strip()) for x in line.strip().split(',') if x.strip()]
            if values:
                labels_list.extend(values)

        if len(labels_list) == 0:
            raise ValueError("Label file is empty")

        if len(labels_list) < n_samples:
            sensor_data = [data[:len(labels_list)] for data in sensor_data]
            n_samples = len(labels_list)
        elif len(labels_list) > n_samples:
            labels_list = labels_list[:n_samples]

        labels = torch.tensor(labels_list, dtype=torch.long)

        all_signals = []
        for data in sensor_data:
            for i in range(data.shape[0]):
                sig = data[i].view(-1)
                all_signals.append(sig)
        all_signals = torch.cat(all_signals, dim=0)

        self._set_adaptive_params(min_dim, max_dim, n_samples, len(sensor_data), all_signals)

        return sensor_data, labels

    def _calculate_quantile(self, sig, q):
        sig_sorted, _ = torch.sort(sig)
        n = sig_sorted.shape[0]
        idx = torch.tensor(q * (n - 1), dtype=torch.float32)
        lower = torch.floor(idx).long()
        upper = torch.ceil(idx).long()

        lower = torch.clamp(lower, 0, n - 1)
        upper = torch.clamp(upper, 0, n - 1)

        if lower == upper:
            return sig_sorted[lower]
        else:
            weight = idx - lower.float()
            return sig_sorted[lower] * (1 - weight) + sig_sorted[upper] * weight

    def _set_adaptive_params(self, min_dim, max_dim, n_samples, n_sensors, all_signals):
        L = min_dim
        N_fp = n_sensors * 1
        self.p = math.ceil(4 * math.log2(max(N_fp, 2)))
        self.p = max(4, min(32, self.p))
        self._set_global_histogram_params(all_signals)

    def _generate_projection_matrix(self, d_in, d_out):
        if self.sparsity_sigma <= 1.0:
            std = 1.0 / math.sqrt(d_out)
            return torch.randn(d_in, d_out) * std
        else:
            prob = 1.0 / self.sparsity_sigma
            mask = torch.rand(d_in, d_out) < prob
            std = 1.0 / math.sqrt(d_out * self.sparsity_sigma)
            values = torch.randn(d_in, d_out) * std
            return mask.float() * values

    def _set_global_histogram_params(self, all_signals):
        n_total = all_signals.shape[0]
        if n_total == 0:
            raise ValueError("No signals available for histogram parameters")

        q1 = self._calculate_quantile(all_signals, 0.25)
        q3 = self._calculate_quantile(all_signals, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        sig_clipped = torch.clamp(all_signals, lower_bound, upper_bound)

        self.global_min = torch.min(sig_clipped)
        self.global_max = torch.max(sig_clipped)
        range_val = self.global_max - self.global_min

        if range_val < 1e-8:
            self.global_k = 10
            return

        h = 2 * iqr / (n_total ** (1 / 3))
        if h < 1e-8:
            h = range_val / 10

        k = torch.ceil(range_val / h).long().item()
        self.global_k = max(8, min(64, k))

    def compute_statistical_features(self, signal):
        if len(signal.shape) == 1:
            sig = signal
        else:
            sig = signal.view(-1)

        n = sig.shape[0]
        if n == 0:
            return torch.ones(self.global_k) / self.global_k

        sig_clipped = torch.clamp(sig, self.global_min, self.global_max)
        hist = torch.histc(sig_clipped, bins=self.global_k, min=self.global_min, max=self.global_max)

        hist_sum = torch.sum(hist)
        if hist_sum > 1e-8:
            hist = hist / hist_sum
        else:
            hist = torch.ones_like(hist) / self.global_k

        return hist

    def _time_series_to_pointcloud(self, signal):
        n_points = signal.shape[0]
        if n_points == 0:
            return torch.zeros(0, 2)

        time_coord = torch.linspace(0, 1, n_points).view(-1, 1)
        pointcloud = torch.cat([time_coord, signal.view(-1, 1)], dim=1)
        return pointcloud

    def _compute_singular_values(self, pointcloud, R):
        if pointcloud.shape[0] == 0:
            return torch.zeros(self.p, device=pointcloud.device)

        Y = pointcloud @ R

        try:
            U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        except RuntimeError:
            return torch.zeros(self.p, device=pointcloud.device)

        n_sv = S.shape[0]
        if n_sv >= self.p:
            return S[:self.p]
        else:
            padded = torch.zeros(self.p, device=S.device)
            padded[:n_sv] = S
            return padded

    def create_single_graph(self, sample_sensor_data):
        n_nodes = len(sample_sensor_data)

        if n_nodes == 0:
            raise ValueError("No sensor data in sample")

        pointclouds = []
        for signal in sample_sensor_data:
            pc = self._time_series_to_pointcloud(signal)
            pointclouds.append(pc)

        R = self._generate_projection_matrix(d_in=2, d_out=self.p)

        sv_features = torch.zeros(n_nodes, self.p)
        for i in range(n_nodes):
            pc = pointclouds[i]
            sv = self._compute_singular_values(pc, R)
            sv_features[i] = sv

        dist_matrix = torch.cdist(sv_features, sv_features, p=2)
        upper_tri = dist_matrix[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) == 1]

        if upper_tri.numel() > 0:
            sigma = torch.median(upper_tri)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        consistency_matrix = torch.exp(-dist_matrix.pow(2) / (2 * sigma ** 2))

        upper_tri_cons = consistency_matrix[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) == 1]
        if upper_tri_cons.numel() > 0:
            mu_cons = torch.mean(upper_tri_cons)
            sigma_cons = torch.std(upper_tri_cons)
            threshold = mu_cons + 0.5 * sigma_cons
        else:
            threshold = 0.1

        adj_matrix = (consistency_matrix >= threshold).float()
        adj_matrix.fill_diagonal_(1.0)

        weight_matrix = consistency_matrix * adj_matrix
        row_sums = weight_matrix.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums < 1e-8, torch.ones_like(row_sums), row_sums)
        weight_matrix = weight_matrix / row_sums

        edge_indices = torch.nonzero(adj_matrix > 0, as_tuple=True)
        if len(edge_indices[0]) == 0:
            nodes = torch.arange(n_nodes)
            edge_index = torch.stack([nodes, nodes], dim=0)
            edge_weights = torch.ones(n_nodes)
        else:
            edge_index = torch.stack([edge_indices[0], edge_indices[1]], dim=0).long()
            edge_weights = weight_matrix[edge_indices]

        node_features_list = []
        for signal in sample_sensor_data:
            stat_feat = self.compute_statistical_features(signal)
            node_features_list.append(stat_feat)

        node_features = torch.stack(node_features_list, dim=0)

        return edge_index, edge_weights, node_features, threshold

    def create_graph_dataset(self, sensor_files, label_file):
        start_time = time.time()

        all_sensor_data, labels = self.load_data(sensor_files, label_file)
        n_samples = len(labels)
        n_sensors = len(all_sensor_data)

        graph_list = []
        total_threshold = 0.0
        valid_count = 0

        for i in range(n_samples):
            try:
                sample_data = [sensor_data[i] for sensor_data in all_sensor_data]
                has_empty_signal = any(
                    len(signal) == 0 if len(signal.shape) == 1 else signal.numel() == 0
                    for signal in sample_data
                )
                if has_empty_signal:
                    continue

                edge_index, edge_attr, node_features, threshold = self.create_single_graph(sample_data)
                total_threshold += threshold

                graph = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=labels[i].long()
                )
                graph_list.append(graph)
                valid_count += 1
            except Exception:
                continue

        if valid_count == 0:
            raise RuntimeError("No valid graphs created")

        creation_time = time.time() - start_time
        self.graph_list = graph_list

        return graph_list, creation_time

    def save_graph_data(self, graph_list, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(graph_list, save_path)

    def load_graph_data(self, save_path):
        if os.path.exists(save_path):
            return torch.load(save_path)
        else:
            raise FileNotFoundError(f"File {save_path} not found")


class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = torch.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor-files', nargs='+', required=True,
                        help='List of sensor CSV files')
    parser.add_argument('--label-file', required=True,
                        help='Label CSV file')
    parser.add_argument('--save-path', default='graph_data.pt',
                        help='Path to save the generated graph dataset')
    parser.add_argument('--sparsity-sigma', type=float, default=1.0,
                        help='Sparsity parameter sigma for random projection')
    args = parser.parse_args()

    creator = AdaptiveSingleScaleGeometricConsistencyGraphCreator(
        embedding_dim=32,
        sparsity_sigma=args.sparsity_sigma
    )

    if os.path.exists(args.save_path):
        graph_list = creator.load_graph_data(args.save_path)
    else:
        graph_list, _ = creator.create_graph_dataset(
            args.sensor_files,
            args.label_file
        )
        creator.save_graph_data(graph_list, args.save_path)

    return graph_list


if __name__ == "__main__":
    main()