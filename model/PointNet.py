import torch
import torch.nn as nn
import torch.nn.functional as F

scaling = 1.0  # reasonable choices for scaling: 4.0, 2.0, 1.0, 0.25, 0.125

class MaxPoolExpDim(nn.Module):
    def __init__(self, num_points):
        super(MaxPoolExpDim, self).__init__()
        self.num_points = num_points

    def forward(self, global_feature):
        global_feature = torch.max(global_feature, 2, keepdim=True)[0]
        global_feature = global_feature.repeat(1, 1, self.num_points)
        return global_feature

class TNet(nn.Module):
    def __init__(self, k, scaling=1.0):
        super(TNet, self).__init__()
        self.k = k
        self.scaling = scaling

        # Shared MLP layers
        self.mlp = nn.Sequential(
            nn.Conv1d(k, int(64 * scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(64 * scaling)),
            nn.Conv1d(int(64 * scaling), int(128 * scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(128 * scaling)),
            nn.Conv1d(int(128 * scaling), int(1024 * scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(1024 * scaling))
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(int(1024 * scaling), int(k * k)),
            #nn.ReLU(),
            #nn.BatchNorm1d(int(512 * scaling)),
            #nn.Linear(int(512 * scaling), int(256 * scaling)),
            #nn.ReLU(),
            #nn.BatchNorm1d(int(256 * scaling)),
            #nn.Linear(int(256 * scaling), k * k)
        )

        # Initialize the last layer's weights with identity matrix
        self.fc[-1].weight.data = torch.eye(k).view(-1, k * k)
        self.fc[-1].bias.data.zero_()

    def forward(self, x):
        B = x.size(0)
        x = self.mlp(x)
        x = torch.max(x, 2)[0]
        x = self.fc(x)
        identity = torch.eye(self.k, device=x.device).unsqueeze(0)
        matrix = x.view(-1, self.k, self.k) + identity
        return matrix


class PointNet(nn.Module):
    def __init__(self, point_numbers, space_variable, n_features, scaling=1.0):
        super(PointNet, self).__init__()
        
        self.input_transform = TNet(k=space_variable, scaling=scaling) # For input spatial coordinates
        self.feature_transform = TNet(k=int(128 * scaling), scaling=scaling) # For features
        
        # Shared MLP (B, 3, P) -> (B, 64, P)
        self.branch1 = nn.Sequential( # 3-> 64
            nn.Conv1d(space_variable, int(64 * scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(64 * scaling)),
            nn.Conv1d(int(64 * scaling), int(128 * scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(128 * scaling))
        )

        # Shared MLP (B, 64, P) -> (B, 1024, P)
        self.branch2 = nn.Sequential(
            nn.Conv1d(int(128 * scaling), int(512*scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(512*scaling)),
            nn.Conv1d(int(512*scaling), int(1024*scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(1024*scaling)),
            nn.Conv1d(int(1024*scaling), int(1024*scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(1024 * scaling))
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(int(1024 * scaling), int(1024*2*scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(1024*2*scaling)),
            nn.Conv1d(int(1024*2*scaling), int(1024*4*scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(1024*4*scaling)),
            nn.Conv1d(int(1024*4*scaling), int(1024*8*scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(1024*8*scaling))
        )

        # Max function
        self.global_feature = nn.MaxPool1d(point_numbers) # Picking the maximum value of each feature

        self.max_pool_exp_dim = MaxPoolExpDim(point_numbers) # Giving the same global features to all the points

        # Shared MLP (B, 1088, P) -> (B, 128, P)
        self.branch3obs = nn.Sequential( #obsolete for now
            nn.Conv1d(int(1088 * scaling), int(1024 * scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(1024 * scaling)),
            nn.Conv1d(int(1024 * scaling), int(1024 * scaling), kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(1024 * scaling)),
            nn.Conv1d(int(1024 * scaling), n_features, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(int(n_features))
        )


        # PointNet output (B, 128, P) -> (B, 3, P)
        
    def forward(self, x):

        x = x.transpose(1, 2)

        # Input transform
        #trans_input = self.input_transform(x)
        #x = torch.bmm(x, trans_input)

        branch1_output = self.branch1(x)
        # Feature transform
        #trans_feature = self.feature_transform(branch1_output)
        #branch1_output = torch.bmm(branch1_output, trans_feature)
                
        branch2_output = self.branch2(branch1_output)
        branch3_output = self.branch3(branch2_output)
        #global_feature = self.global_feature(branc30_output)
        #global_feature = self.max_pool_exp_dim(global_feature)
        global_feature = torch.max(branch3_output, 2, keepdim=True)[0]

        #concatenated_features = torch.cat((local_feature, global_feature), dim=1)
        
        #branch3_output = self.branch3obs(concatenated_features)
        #global_feature = global_feature.view(global_feature.size(0), -1, 1)
        #print(global_feature.shape)
        global_feature = global_feature.view(global_feature.shape[0], 512, 16)
        pointnet_output = global_feature.transpose(1, 2)
        return pointnet_output