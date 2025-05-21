import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# Define the hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
TIME_STEPS = 12  # Input sequence length
PREDICT_STEPS = 1  # Predict next step
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHT_DECAY = 0.0001

class ChebConv(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        Chebyshev graph convolution operation
        """
        super(ChebConv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, 0)
    
    def forward(self, x):
        """
        :param x: Input data of shape [batch_size, num_nodes, in_channels]
        :return: Output data of shape [batch_size, num_nodes, out_channels]
        """
        batch_size, num_nodes, in_channels = x.shape
        x = x.permute(0, 2, 1).contiguous()  # [batch_size, in_channels, num_nodes]
        
        outputs = []
        for k in range(self.K):
            T_k = self.cheb_polynomials[k]  # [num_nodes, num_nodes]
            T_k = T_k.to(x.device)
            T_k_x = torch.matmul(x, T_k)  # [batch_size, in_channels, num_nodes]
            T_k_x = T_k_x.permute(0, 2, 1)  # [batch_size, num_nodes, in_channels]
            
            # Apply weights for each filter order
            theta_k = self.weights[k]  # [in_channels, out_channels]
            T_k_x_theta = torch.matmul(T_k_x, theta_k)  # [batch_size, num_nodes, out_channels]
            outputs.append(T_k_x_theta)
            
        # Sum outputs and add bias
        out = torch.sum(torch.stack(outputs, dim=0), dim=0) + self.bias
        return out

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        
    def forward(self, x):
        """Gated TCN: GLU(X) = (X * sigmoid(Conv(X))) + tanh(Conv(X))"""
        # x shape: [batch_size, time_steps, num_nodes, channels]
        x = x.permute(0, 3, 2, 1)  # [batch_size, channels, num_nodes, time_steps]
        
        p = self.conv1(x)  # Shape: [batch_size, channels, num_nodes, time_steps-kernel_size+1]
        q = torch.sigmoid(self.conv2(x))
        pq = p * q
        
        r = torch.tanh(self.conv3(x))  # Residual connection
        out = pq + r  # Element-wise addition
        
        # Back to original shape
        out = out.permute(0, 3, 2, 1)  # [batch_size, time_steps-kernel_size+1, num_nodes, channels]
        return out

class STConvBlock(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, temporal_channels, spatial_channels, num_nodes):
        super(STConvBlock, self).__init__()
        self.temporal_conv1 = TemporalConvLayer(in_channels, temporal_channels)
        self.cheb_conv = ChebConv(K, cheb_polynomials, temporal_channels, spatial_channels)
        self.temporal_conv2 = TemporalConvLayer(spatial_channels, spatial_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch_size, time_steps, num_nodes, in_channels]
        out = self.temporal_conv1(x)  # [batch_size, time_steps-2, num_nodes, temporal_channels]
        
        # Apply spatial graph convolution to each time step
        batch_size, time_steps, num_nodes, channels = out.shape
        out_list = []
        
        for t in range(time_steps):
            out_t = self.cheb_conv(out[:, t, :, :])  # [batch_size, num_nodes, spatial_channels]
            out_list.append(out_t.unsqueeze(1))  # Add time dimension back
            
        out = torch.cat(out_list, dim=1)  # [batch_size, time_steps, num_nodes, spatial_channels]
        
        # Apply second temporal convolution
        out = self.temporal_conv2(out)  # [batch_size, time_steps-4, num_nodes, spatial_channels]
        
        # Apply layer norm and ReLU
        out = out.permute(0, 2, 1, 3)  # [batch_size, num_nodes, time_steps-4, spatial_channels]
        out = self.batch_norm(out)  # Apply batch norm
        out = out.permute(0, 2, 1, 3)  # [batch_size, time_steps-4, num_nodes, spatial_channels]
        return self.relu(out)

class OutputLayer(nn.Module):
    def __init__(self, in_channels, output_steps, num_nodes):
        super(OutputLayer, self).__init__()
        self.fc = nn.Linear(in_channels, output_steps)
        
    def forward(self, x):
        # x shape: [batch_size, time_steps, num_nodes, in_channels]
        batch_size, time_steps, num_nodes, channels = x.shape
        
        out = x.reshape(batch_size, time_steps, num_nodes * channels)
        out = self.fc(out)  # [batch_size, time_steps, num_nodes * output_steps]
        
        # Reshape to [batch_size, output_steps, num_nodes, 1]
        out = out.reshape(batch_size, time_steps, num_nodes, -1)
        return out

class STGCN(nn.Module):
    def __init__(self, K, cheb_polynomials, num_nodes, input_steps, output_steps=1):
        super(STGCN, self).__init__()
        
        # Model architecture parameters
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_nodes = num_nodes
        
        # Model layers
        self.st_block1 = STConvBlock(K, cheb_polynomials, 1, 64, 32, num_nodes)
        self.st_block2 = STConvBlock(K, cheb_polynomials, 32, 64, 128, num_nodes)
        
        # Output layer
        self.output_layer = OutputLayer(128, output_steps, num_nodes)
        
    def forward(self, x):
        # x shape: [batch_size, time_steps, num_nodes, channels]
        out = self.st_block1(x)  # Temporal kernel size 3 (twice): time_steps -> time_steps-4
        out = self.st_block2(out)  # Temporal kernel size 3 (twice): (time_steps-4) -> (time_steps-8)
        out = self.output_layer(out[:, -1:, :, :])  # Use the last time step output
        
        return out[:, 0, :, :]  # Return [batch_size, num_nodes, output_steps]

def compute_cheb_polynomials(L, K):
    """
    Compute Chebyshev polynomials up to order K.
    """
    n = L.shape[0]  # Number of nodes
    laplacian_list = [np.eye(n), L]  # T_0, T_1
    
    for k in range(2, K):
        laplacian_list.append(2 * L @ laplacian_list[k-1] - laplacian_list[k-2])
    
    return [torch.from_numpy(i).float() for i in laplacian_list]

def construct_adj_matrix(csv_path, num_nodes):
    """
    Construct adjacency matrix from CSV file containing edges and distances.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize adjacency matrix with zeros
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    # Fill adjacency matrix with distances
    for _, row in df.iterrows():
        from_node = int(row['from'])
        to_node = int(row['to'])
        distance = float(row['distance'])
        
        # Create a weighted adjacency matrix where weight = 1/distance
        # Prevents division by zero by adding a small epsilon
        weight = 1.0 / (distance + 1e-6)
        adj_matrix[from_node, to_node] = weight
        adj_matrix[to_node, from_node] = weight  # Assume undirected graph
    
    # Make sure diagonal is 0
    np.fill_diagonal(adj_matrix, 0)
    
    return adj_matrix

def compute_laplacian(adj_matrix):
    """
    Compute normalized Laplacian matrix: L = I - D^(-1/2) * A * D^(-1/2)
    """
    # Calculate degree matrix
    degree_matrix = np.sum(adj_matrix, axis=1)
    
    # Calculate D^(-1/2)
    degree_matrix_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix + 1e-6))
    
    # Calculate normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
    norm_adj = degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt
    
    # Calculate Laplacian matrix
    laplacian = np.eye(adj_matrix.shape[0]) - norm_adj
    
    return laplacian

class TrafficDataset(Dataset):
    def __init__(self, data, time_steps, pred_steps, scaler=None, train=True):
        self.time_steps = time_steps
        self.pred_steps = pred_steps
        self.train = train
        
        # Normalize data if in training mode
        if train:
            if scaler is None:
                self.scaler = StandardScaler()
                data = self.scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
            else:
                self.scaler = scaler
                data = self.scaler.transform(data.reshape(-1, 1)).reshape(data.shape)
        else:
            self.scaler = scaler
            if scaler is not None:
                data = self.scaler.transform(data.reshape(-1, 1)).reshape(data.shape)
        
        self.data = data
        self.samples = self.prepare_samples()
    
    def prepare_samples(self):
        samples = []
        for i in range(len(self.data) - self.time_steps - self.pred_steps + 1):
            x = self.data[i:i+self.time_steps]
            y = self.data[i+self.time_steps:i+self.time_steps+self.pred_steps]
            samples.append((x, y))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def get_scaler(self):
        return self.scaler

def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x = x.to(device)  # [batch_size, time_steps, num_nodes, channels]
            y = y.to(device)  # [batch_size, pred_steps, num_nodes, channels]
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y.squeeze(3))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate the model
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                y_pred = model(x)
                loss = loss_fn(y_pred, y.squeeze(3))
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
        
        print(f"Epoch \{epoch+1}/\{epochs}, Train Loss: \{train_loss:.4f\}, Val Loss: \{val_loss:.4f\}")
    
    # Load best model
    model.load_state_dict(best_model)
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, scaler, device):
    model.eval()
    mae_sum = 0
    mse_sum = 0
    samples = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            
            # Convert to numpy for evaluation
            y_pred_np = y_pred.cpu().numpy()
            y_true_np = y.squeeze(3).cpu().numpy()
            
            # Inverse transform if scaler is provided
            if scaler is not None:
                y_pred_np = scaler.inverse_transform(y_pred_np.reshape(-1, 1)).reshape(y_pred_np.shape)
                y_true_np = scaler.inverse_transform(y_true_np.reshape(-1, 1)).reshape(y_true_np.shape)
            
            # Calculate metrics
            mae = mean_absolute_error(y_true_np.reshape(-1), y_pred_np.reshape(-1))
            mse = mean_squared_error(y_true_np.reshape(-1), y_pred_np.reshape(-1))
            
            mae_sum += mae * len(x)
            mse_sum += mse * len(x)
            samples += len(x)
    
    mae = mae_sum / samples
    rmse = np.sqrt(mse_sum / samples)
    mape = np.mean(np.abs((y_true_np.reshape(-1) - y_pred_np.reshape(-1)) / (y_true_np.reshape(-1) + 1e-5))) * 100
    
    return mae, rmse, mape

def main():
    # Load data
    npz_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.npz"
    csv_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.csv"
    
    data_obj = np.load(npz_path)
    traffic_data = data_obj['data']  # Shape (26208, 358, 1)
    
    time_steps = traffic_data.shape[0]
    num_nodes = traffic_data.shape[1]
    num_features = traffic_data.shape[2]
    
    print(f"Data shape: \{traffic_data.shape}")
    print(f"Time steps: \{time_steps}")
    print(f"Number of nodes: \{num_nodes}")
    print(f"Number of features: \{num_features}")
    
    # Construct adjacency matrix
    adj_matrix = construct_adj_matrix(csv_path, num_nodes)
    
    # Compute Laplacian and Chebyshev polynomials
    laplacian = compute_laplacian(adj_matrix)
    K = 3  # Chebyshev polynomial order
    cheb_polynomials = compute_cheb_polynomials(laplacian, K)
    
    # Split dataset into train, validation and test
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    
    train_size = int(time_steps * train_ratio)
    val_size = int(time_steps * val_ratio)
    
    train_data = traffic_data[:train_size]
    val_data = traffic_data[train_size:train_size+val_size]
    test_data = traffic_data[train_size+val_size:]
    
    # Create datasets
    input_steps = 12
    pred_steps = 1
    
    train_dataset = TrafficDataset(train_data, input_steps, pred_steps)
    scaler = train_dataset.get_scaler()
    val_dataset = TrafficDataset(val_data, input_steps, pred_steps, scaler=scaler, train=False)
    test_dataset = TrafficDataset(test_data, input_steps, pred_steps, scaler=scaler, train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = STGCN(K, cheb_polynomials, num_nodes, input_steps, pred_steps).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, loss_fn, EPOCHS, DEVICE
    )
    training_time = time.time() - start_time
    print(f"Training finished in \{training_time:.2f} seconds")
    
    # Evaluate model
    print("Evaluating model on test set...")
    mae, rmse, mape = evaluate_model(model, test_loader, scaler, DEVICE)
    
    print(f"Test MAE: \{mae:.4f}")
    print(f"Test RMSE: \{rmse:.4f}")
    print(f"Test MAPE: \{mape:.4f}%")

if __name__ == "__main__":
    main()