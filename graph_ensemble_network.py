import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
from torch_geometric.nn import ChebConv, global_mean_pool


class FeatureAugmentor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.augmentation = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.augmentation(x)


class GraphEnsembleNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, L=3, C=2):
        super().__init__()
        self.L = L
        self.C = C
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        if input_dim != hidden_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        else:
            self.input_proj = nn.Identity()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.feature_augmentors = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(L):
            layer_convs = nn.ModuleList()
            layer_bns = nn.ModuleList()
            layer_augmentors = nn.ModuleList()
            layer_norms = nn.ModuleList()

            for _ in range(C):
                layer_convs.append(ChebConv(hidden_dim, hidden_dim, K=2))
                layer_bns.append(nn.BatchNorm1d(hidden_dim))
                layer_augmentors.append(FeatureAugmentor(hidden_dim))
                layer_norms.append(nn.LayerNorm(hidden_dim))

            self.convs.append(layer_convs)
            self.bns.append(layer_bns)
            self.feature_augmentors.append(layer_augmentors)
            self.layer_norms.append(layer_norms)

        self.ensemble_weights = nn.Parameter(torch.ones(L, C) / (L * C))

        self.pool_process = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim // 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data, return_features=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)

        h = x
        for layer in range(self.L):
            h_outputs = []

            for classifier in range(self.C):
                h_aug = self.feature_augmentors[layer][classifier](h)
                h_c = self.convs[layer][classifier](h_aug, edge_index)
                h_c = self.bns[layer][classifier](h_c)
                h_c = F.relu(h_c)
                h_c = self.layer_norms[layer][classifier](h_c)

                if h_c.size(1) == h.size(1):
                    h_c = 0.9 * h_c + 0.1 * h

                h_outputs.append(h_c)

            weights = F.softmax(self.ensemble_weights[layer], dim=0)
            h = torch.zeros_like(h_outputs[0])
            for i, h_out in enumerate(h_outputs):
                h += weights[i] * h_out

            h = F.layer_norm(h, h.shape[1:])

        graph_emb = global_mean_pool(h, batch)
        graph_emb = self.pool_process(graph_emb)
        logits = self.classifier(graph_emb)

        if return_features:
            return logits, graph_emb
        return logits


def load_graph_dataset(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    return torch.load(data_path)


def train_model(model, train_loader, val_loader, device, epochs=100):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for _ in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, data.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            pred = output.argmax(dim=1)
            train_correct += pred.eq(data.y).sum().item()
            train_total += data.y.size(0)
            total_loss += loss.item() * data.num_graphs

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_loss = total_loss / train_total if train_total > 0 else 0.0
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0.0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data.y)

                pred = output.argmax(dim=1)
                val_correct += pred.eq(data.y).sum().item()
                val_total += data.y.size(0)
                val_loss_total += loss.item() * data.num_graphs

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_loss_total / val_total if val_total > 0 else 0.0
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        scheduler.step(val_acc)

    return {
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def perform_cross_validation(graph_list, device, epochs=100, num_folds=5):
    labels = [int(g.y.item()) for g in graph_list]

    label_groups = {}
    for idx, label in enumerate(labels):
        label_groups.setdefault(label, []).append(idx)

    fold_groups = []
    for _, indices in label_groups.items():
        kfold = KFold(n_splits=num_folds, shuffle=True)
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            if len(fold_groups) <= fold_idx:
                fold_groups.append({'train': [], 'val': []})
            fold_groups[fold_idx]['train'].extend([indices[i] for i in train_idx])
            fold_groups[fold_idx]['val'].extend([indices[i] for i in val_idx])

    fold_results = []

    for fold in range(num_folds):
        train_graphs = [graph_list[i] for i in fold_groups[fold]['train']]
        val_graphs = [graph_list[i] for i in fold_groups[fold]['val']]

        batch_size = min(16, len(train_graphs))
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

        input_dim = train_graphs[0].x.size(1)
        hidden_dim = 64
        num_classes = len(set(labels))

        model = GraphEnsembleNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            L=2,
            C=2
        ).to(device)

        history = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs
        )

        final_val_acc = history['val_accuracies'][-1] if history['val_accuracies'] else 0.0
        final_train_acc = history['train_accuracies'][-1] if history['train_accuracies'] else 0.0

        fold_results.append({
            'fold': fold + 1,
            'final_val_acc': final_val_acc,
            'final_train_acc': final_train_acc,
            'val_accuracies': history['val_accuracies'],
            'train_accuracies': history['train_accuracies'],
            'val_losses': history['val_losses'],
            'train_losses': history['train_losses']
        })

    return fold_results


def main(data_path, epochs=100, num_folds=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_list = load_graph_dataset(data_path)
    fold_results = perform_cross_validation(
        graph_list,
        device,
        epochs=epochs,
        num_folds=num_folds
    )

    final_val_accs = [result['final_val_acc'] for result in fold_results]
    avg_val_acc = sum(final_val_accs) / len(final_val_accs)
    std_val_acc = (
        sum((x - avg_val_acc) ** 2 for x in final_val_accs) / len(final_val_accs)
    ) ** 0.5

    return fold_results, avg_val_acc, std_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    main(args.data, epochs=args.epochs, num_folds=args.folds)