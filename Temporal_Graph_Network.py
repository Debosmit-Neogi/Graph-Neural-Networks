import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

'''
Problem: Given a small social media with three people: A,B,C

At t1: A interacts with B.
At t2: B interacts with C.
At t3: A interacts with C


GOAL: After t3, Predict the probability of A interacting with B
'''

# Define Temporal GNN Model
class TemporalGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(TemporalGNN, self).__init__(aggr='mean')  # Mean aggregation
        self.lin = nn.Linear(in_channels + 1, out_channels)  # +1 for time encoding
        self.act = nn.ReLU()
        self.gru = nn.GRU(out_channels, hidden_dim)
        self.memory = torch.zeros(3, hidden_dim)  # Initialize memory for 3 nodes
    
    def forward(self, x, edge_index, edge_attr):
        for t in range(len(edge_attr) // 2):  # Process events sequentially
            messages = self.propagate(edge_index[:, t * 2: (t + 1) * 2], x=x, edge_attr=edge_attr[t * 2: (t + 1) * 2])
            
            # Ensure correct GRU input shape: (1, batch_size, hidden_dim)
            self.memory = self.memory.detach().unsqueeze(0)  # Convert (batch_size, hidden_dim) -> (1, batch_size, hidden_dim)
            self.memory, _ = self.gru(messages.unsqueeze(0), self.memory)
            self.memory = self.memory.squeeze(0)  # Convert back after GRU

        return self.memory

    
    def message(self, x_j, edge_attr):
        return self.act(self.lin(torch.cat([x_j, edge_attr], dim=1)))

# Define Decoder Network
class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z_a, z_b):
        combined = torch.cat([z_a, z_b], dim=-1)
        return self.sigmoid(self.fc(combined))

# Define Nodes and Temporal Edges
timestamps = torch.tensor([1.0, 2.0, 3.0]).view(-1, 1)  # Time encoding
nodes = torch.eye(3)  # One-hot encoding for A, B, C (3 nodes)

t1_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).T  # A ↔ B
t2_edges = torch.tensor([[1, 2], [2, 1]], dtype=torch.long).T  # B ↔ C
t3_edges = torch.tensor([[0, 2], [2, 0]], dtype=torch.long).T  # A ↔ C

temporal_edges = torch.cat([t1_edges, t2_edges, t3_edges], dim=1)
temporal_times = torch.cat([timestamps[0].repeat(2,1), timestamps[1].repeat(2,1), timestamps[2].repeat(2,1)])

data = Data(x=nodes, edge_index=temporal_edges, edge_attr=temporal_times)

# Model Initialization
hidden_dim = 3
model = TemporalGNN(in_channels=3, out_channels=2, hidden_dim=hidden_dim)
decoder = Decoder(hidden_dim=hidden_dim)
optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

def predict_interaction(model, decoder, node_a, node_b):
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, data.edge_attr)
        prob = decoder(embeddings[node_a], embeddings[node_b])
        return prob.item()

# Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out, data.x)  # Dummy loss (can use actual supervised labels)
    loss.backward(loss.backward(retain_graph=True))
    optimizer.step()

# Compute final node embeddings from memory
Z_A, Z_B, Z_C = model.memory[0], model.memory[1], model.memory[2]

# Compute probability using decoder
prob_A_B = predict_interaction(model, decoder, 0, 1)
print(f"Predicted probability of A interacting with B after t3: {prob_A_B:.4f}")
