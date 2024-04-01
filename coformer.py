import torch
import torch.nn as nn

class Coformer(nn.Module):
    def __init__(self, pool_method=None, stats=None, d_model=256, nhead=4, npoints=5, num_encoder_layers=2,
                dim_feedforward=512, pose_dim=7, feat_dim=6528, pos_dim=2, action_dim=7,
                dropout=0.1, avg_pool=True, random_drop=None, invalid_mask=False, drop_spectrum=False):
        super(Coformer, self).__init__()
        self.d_model = d_model
        self.drop_spectrum = drop_spectrum
        self.pose_dim = pose_dim
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim

        self.pose_linear = nn.Linear(pose_dim, d_model)
        self.position_linear = nn.Linear(pos_dim, d_model // 2)
        self.feature_linear = nn.Linear(feat_dim, d_model // 2)

        # Shared positional encodings for P, X, and L
        self.P_embedding = nn.Parameter(torch.randn(1, d_model))
        self.X_embedding = nn.Parameter(torch.randn(1, d_model // 2))
        self.L_embedding = nn.Parameter(torch.randn(1, d_model // 2))

        # Positional encodings for each P
        self.corr_embedding = nn.Embedding(npoints, d_model)

        # Mask tokens for randomly dropping points
        self.X_mask_token = nn.Parameter(torch.randn(d_model // 2))
        self.L_mask_token = nn.Parameter(torch.randn(d_model // 2))
        self.invalid_mask = invalid_mask
        if invalid_mask:
            self.invalid_token = nn.Parameter(torch.randn(d_model // 2))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layers
        self.output_average = nn.AdaptiveAvgPool1d(1)
        self.output_linear = nn.Linear(d_model, action_dim)
        self.action_stats = stats


        # so far avg no pose has been the best, gripper value sucks tho
        # avg also is ok (but gripper value bad)
        # valid no pose still doesn't go right far enough if it starts from the left
        # valid gripper value sucks
        self.pool_method = pool_method
        if pool_method is None: # for backwards compatibility with older checkpoints
            if avg_pool == True:
                self.pool_method = "avg"
            else:
                print("WARNING: avg_pool is False, but pool_method is None. Setting pool_method to valid_no_pose")
                self.pool_method = "valid_no_pose"
        print("******Using pool_method: {}".format(self.pool_method))
        self.npoints = npoints
        self.random_drop = random_drop
        self.drop_spectrum = drop_spectrum

    def forward(self, poses, features, positions):
        """
        Input:
        poses: (N, 7)
        positions: (N, S, 2) with invalid as (-2,-2)
        features: (N, S, 6528) 
        """
        N, S = positions.shape[:2]
        # Check input dimensions
        assert poses.dim() == 2 and poses.shape[-1] == self.pose_dim, "Invalid shape for pose"
        assert features.dim() == 3 and features.shape[-1] == self.feat_dim and features.shape[-2] == self.npoints, "Invalid shape for features, {}".format(features.shape)
        assert positions.dim() == 3 and positions.shape[-1] == self.pos_dim and positions.shape[-2] == self.npoints, "Invalid shape for positions"

        invalid_mask = (positions == torch.tensor([-2, -2], device=positions.device)).all(-1, keepdim=True)
        # Apply linear transformations
        poses = self.pose_linear(poses).unsqueeze(1)  # (N, 1, d_model)
        features = self.feature_linear(features)  # (N, S, d_model)
        positions = self.position_linear(positions)  # (N, S, d_model)

        if self.invalid_mask:
            # Apply invalid mask
            positions = torch.where(invalid_mask, self.invalid_token, positions)
            features = torch.where(invalid_mask, self.invalid_token, features)

        if self.random_drop is not None:
            random_drop_rate = self.random_drop
            if self.drop_spectrum:
                random_drop_rate = torch.rand(1, device=features.device) * self.random_drop
            mask = torch.rand(self.npoints, device=features.device) < random_drop_rate
            features_mask = mask.unsqueeze(-1).expand_as(features)
            positions_mask = mask.unsqueeze(-1).expand_as(positions)
            features = torch.where(features_mask, self.X_mask_token, features)
            positions = torch.where(positions_mask, self.L_mask_token, positions)

        # Add positional embeddings
        poses_embed = poses + self.P_embedding
        feats_embed = features + self.X_embedding
        positions_embed = positions + self.L_embedding

        # Expand and add correlation embeddings
        # corr_embed = self.corr_embedding(torch.arange(S, device=poses.device)).expand(N, -1, -1)
        # feats_embed = feats_embed + corr_embed
        # positions_embed = positions_embed + corr_embed

        combined_features_positions = torch.cat([feats_embed, positions_embed], dim=-1)

        # Combine inputs
        combined_input = torch.cat([
            poses_embed, 
            combined_features_positions
        ], dim=1)  # (N, 1 + 2S, d_model)
    
        if torch.isnan(combined_input).any():
            raise ValueError("NaN values found in combined_input")
        

        # Transformer Encoder
        transformer_output = self.transformer_encoder(combined_input)

        if self.pool_method == "avg":
            output = self.output_average(transformer_output.permute(0, 2, 1)).squeeze(-1)
        elif self.pool_method == "avg_no_pose":
            output = self.output_average(transformer_output[:, 1:].permute(0, 2, 1)).squeeze(-1)
        elif self.pool_method == "valid":
            extended_invalid_mask = torch.cat([torch.ones_like(invalid_mask[:, :1]), invalid_mask, invalid_mask], dim=1)  # (N, 2S, 1)
            # Create a mask for valid data points
            valid_mask = ~extended_invalid_mask.squeeze(-1)  # (N, 2S), True for valid data points

            # Use valid_mask to filter transformer_output
            # valid_outputs = transformer_output[:, 1:]  # exclude pose embedding
            masked_output = torch.where(valid_mask.unsqueeze(-1), transformer_output, torch.tensor(0., device=transformer_output.device))
            sum_output = masked_output.sum(dim=1)
            
            # Calculate the number of valid data points
            num_valid = valid_mask.sum(dim=1, keepdim=True)
            # Prevent division by zero
            num_valid = torch.where(num_valid == 0, torch.tensor(1, device=num_valid.device), num_valid)
            # Calculate average
            output = sum_output / num_valid.float()
        elif self.pool_method == "valid_no_pose":
            extended_invalid_mask = torch.cat([invalid_mask, invalid_mask], dim=1)  # (N, 2S, 1)
            # Create a mask for valid data points
            valid_mask = ~extended_invalid_mask.squeeze(-1)  # (N, 2S), True for valid data points

            # Use valid_mask to filter transformer_output
            everything_but_pose = transformer_output[:, 1:]  # exclude pose embedding
            masked_output = torch.where(valid_mask.unsqueeze(-1), everything_but_pose, torch.tensor(0., device=everything_but_pose.device))
            sum_output = masked_output.sum(dim=1)
            
            # Calculate the number of valid data points
            num_valid = valid_mask.sum(dim=1, keepdim=True)
            # Prevent division by zero
            num_valid = torch.where(num_valid == 0, torch.tensor(1, device=num_valid.device), num_valid)
            # Calculate average
            output = sum_output / num_valid.float()

        action = self.output_linear(output)
        return action

    def denorm_action(self, action):
        mean = self.action_stats['action'][0]
        std = self.action_stats['action'][1]
        return action * mean + std
    
    def normalize_input(self, poses, positions, features):
        mean = self.action_stats['pose'][0]
        std = self.action_stats['pose'][1]
        poses = (poses - mean) / std

        mean = self.action_stats['positions'][0]
        std = self.action_stats['positions'][1]
        positions = (positions - mean) / std

        mean = self.action_stats['features'][0]
        std = self.action_stats['features'][1]
        features = (features - mean) / std
        return poses, positions, features
        
    def step(self, poses, positions, features, use_features=True, use_positions=True):
        """
        input: image sequence, action sequence
        """
        self.eval()
        N = poses.shape[0]
        poses, positions, features = self.normalize_input(poses, positions, features)
        if not use_positions:
            print('zeroing out positions!')
            positions = torch.zeros_like(positions)
        if not use_features:
            print('zeroing out features!')
            features = torch.zeros_like(features)
        action = self(poses, features, positions) 
        return self.denorm_action(action).squeeze(), {}


class Pointformer(nn.Module):
    def __init__(self, action_stats=None, d_model=256, nhead=4, npoints=5, num_encoder_layers=2,
                dim_feedforward=512, pose_dim=7, feat_dim=6528, pos_dim=2, action_dim=7,
                dropout=0.1, avg_pool=True, pool_method="avg", random_drop=None, invalid_mask=False):
        super(Pointformer, self).__init__()
        self.d_model = d_model

        self.feat_dim = feat_dim
        self.pos_dim = pos_dim

        self.position_linear = nn.Linear(pos_dim, d_model)
        self.feature_linear = nn.Linear(feat_dim, d_model)

        # Shared positional encodings for P, X, and L
        self.X_embedding = nn.Parameter(torch.randn(1, d_model))
        self.L_embedding = nn.Parameter(torch.randn(1, d_model))

        # Positional encodings for each P
        self.corr_embedding = nn.Embedding(npoints, d_model)

        # Mask tokens for randomly dropping points
        self.X_mask_token = nn.Parameter(torch.randn(d_model))
        self.L_mask_token = nn.Parameter(torch.randn(d_model))
        self.invalid_mask = invalid_mask
        if invalid_mask:
            self.invalid_token = nn.Parameter(torch.randn(d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layers
        self.output_average = nn.AdaptiveAvgPool1d(1)
        self.output_linear = nn.Linear(d_model, action_dim)
        self.action_stats = action_stats
        self.avg_pool = avg_pool
        self.npoints = npoints
        self.random_drop = random_drop

    def encode(self, poses, features, positions):
        """
        Input:
        poses: (N, 7)
        positions: (N, S, 2) with invalid as (-2,-2)
        features: (N, S, 6528) 
        """
        N, S = positions.shape[:2]
        # Check input dimensions
        assert features.dim() == 3 and features.shape[-1] == self.feat_dim and features.shape[-2] == self.npoints, "Invalid shape for features, {}".format(features.shape)
        assert positions.dim() == 3 and positions.shape[-1] == self.pos_dim and positions.shape[-2] == self.npoints, "Invalid shape for positions"

        invalid_mask = (positions == torch.tensor([-2, -2], device=positions.device)).all(-1, keepdim=True)

        # Apply linear transformations
        features = self.feature_linear(features)  # (N, S, d_model)
        positions = self.position_linear(positions)  # (N, S, d_model)

        if self.invalid_mask:
            # Apply invalid mask
            positions = torch.where(invalid_mask, self.invalid_token, positions)
            features = torch.where(invalid_mask, self.invalid_token, features)

        if self.random_drop is not None:
            mask = torch.rand(self.npoints, device=features.device) < self.random_drop
            features_mask = mask.unsqueeze(-1).expand_as(features)
            positions_mask = mask.unsqueeze(-1).expand_as(positions)
            features = torch.where(features_mask, self.X_mask_token, features)
            positions = torch.where(positions_mask, self.L_mask_token, positions)

        # Add positional embeddings
        feats_embed = features + self.X_embedding
        positions_embed = positions + self.L_embedding

        # Expand and add correlation embeddings
        corr_embed = self.corr_embedding(torch.arange(S, device=features.device)).expand(N, -1, -1)
        feats_embed = feats_embed + corr_embed
        positions_embed = positions_embed + corr_embed

        # Combine inputs
        combined_input = torch.cat([
            feats_embed, 
            positions_embed
        ], dim=1)  # (N, 1 + 2S, d_model)
    
        if torch.isnan(combined_input).any():
            raise ValueError("NaN values found in combined_input")

        # Transformer Encoder
        transformer_output = self.transformer_encoder(combined_input)
        return transformer_output

    def forward(self, poses, features, positions):
        """
        Input:
        poses: (N, 7)
        positions: (N, S, 2) with invalid as (-2,-2)
        features: (N, S, 6528) 
        """
        N, S = positions.shape[:2]
        # Check input dimensions
        assert features.dim() == 3 and features.shape[-1] == self.feat_dim and features.shape[-2] == self.npoints, "Invalid shape for features, {}".format(features.shape)
        assert positions.dim() == 3 and positions.shape[-1] == self.pos_dim and positions.shape[-2] == self.npoints, "Invalid shape for positions"

        invalid_mask = (positions == torch.tensor([-2, -2], device=positions.device)).all(-1, keepdim=True)

        # Apply linear transformations
        features = self.feature_linear(features)  # (N, S, d_model)
        positions = self.position_linear(positions)  # (N, S, d_model)

        if self.invalid_mask:
            # Apply invalid mask
            positions = torch.where(invalid_mask, self.invalid_token, positions)
            features = torch.where(invalid_mask, self.invalid_token, features)

        if self.random_drop is not None:
            mask = torch.rand(self.npoints, device=features.device) < self.random_drop
            features_mask = mask.unsqueeze(-1).expand_as(features)
            positions_mask = mask.unsqueeze(-1).expand_as(positions)
            features = torch.where(features_mask, self.X_mask_token, features)
            positions = torch.where(positions_mask, self.L_mask_token, positions)

        # Add positional embeddings
        feats_embed = features + self.X_embedding
        positions_embed = positions + self.L_embedding

        # Expand and add correlation embeddings
        corr_embed = self.corr_embedding(torch.arange(S, device=features.device)).expand(N, -1, -1)
        feats_embed = feats_embed + corr_embed
        positions_embed = positions_embed + corr_embed

        # Combine inputs
        combined_input = torch.cat([
            feats_embed, 
            positions_embed
        ], dim=1)  # (N, 1 + 2S, d_model)
    
        if torch.isnan(combined_input).any():
            raise ValueError("NaN values found in combined_input")

        # Transformer Encoder
        transformer_output = self.transformer_encoder(combined_input)

        if self.avg_pool:
            output = self.output_average(transformer_output.permute(0, 2, 1)).squeeze(-1)
        else: 
            extended_invalid_mask = torch.cat([invalid_mask, invalid_mask], dim=1)  # (N, 2S, 1)
            # Create a mask for valid data points
            valid_mask = ~extended_invalid_mask.squeeze(-1)  # (N, 2S), True for valid data points

            # Use valid_mask to filter transformer_output
            masked_output = torch.where(valid_mask.unsqueeze(-1), transformer_output, torch.tensor(0., device=transformer_output.device))
            sum_output = masked_output.sum(dim=1)
            
            # Calculate the number of valid data points
            num_valid = valid_mask.sum(dim=1, keepdim=True)
            # Prevent division by zero
            num_valid = torch.where(num_valid == 0, torch.tensor(1, device=num_valid.device), num_valid)
            # Calculate average
            output = sum_output / num_valid.float()
        action = self.output_linear(output)
        return action

    def denorm_action(self, action):
        return action * self.action_stats['std'] + self.action_stats['mean']

    def step(self, poses, positions, features, use_features=True, use_positions=True, use_pose=True):
        """
        input: image sequence, action sequence
        """
        self.eval()
        N = features.shape[0]
        if not use_positions:
            print('zeroing out positions!')
            positions = torch.zeros_like(positions)
        if not use_features:
            print('zeroing out features!')
            features = torch.zeros_like(features)

        action = self(poses, features, positions) 

        return self.denorm_action(action).squeeze(), {}



class Poseformer(nn.Module):
    def __init__(self, action_stats=None, d_model=256, nhead=4, npoints=5, num_encoder_layers=2,
                dim_feedforward=512, pose_dim=7, feat_dim=6528, pos_dim=2, action_dim=7,
                dropout=0.1, avg_pool=True, random_drop=None):
        super(Poseformer, self).__init__()
        self.d_model = d_model

        self.pose_dim = pose_dim

        self.pose_linear = nn.Linear(pose_dim, d_model)

        # Shared positional encodings for P, X, and L
        self.P_embedding = nn.Parameter(torch.randn(1, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layers
        self.output_average = nn.AdaptiveAvgPool1d(1)
        self.output_linear = nn.Linear(d_model, action_dim)
        self.action_stats = action_stats
        self.avg_pool = avg_pool
        self.npoints = npoints
        self.random_drop = random_drop

    def forward(self, poses, features, positions):
        """
        Input:
        poses: (N, 7)
        positions: (N, S, 3) with invalid as (-1,-1,-1)
        features: (N, S, 6528) 
        """
        N = poses.shape[0]
        # Check input dimensions
        assert poses.dim() == 2 and poses.shape[-1] == self.pose_dim, "Invalid shape for pose"

        # Apply linear transformations
        poses = self.pose_linear(poses).unsqueeze(1)  # (N, 1, d_model)
        # Transformer Encoder
        transformer_output = self.transformer_encoder(poses)

        if self.avg_pool:
            output = self.output_average(transformer_output.permute(0, 2, 1)).squeeze(-1)
        else: 
            output = transformer_output[:,0,:]
        action = self.output_linear(output)
        return action

    def denorm_action(self, action):
        mean = self.action_stats['mean']
        std = self.action_stats['std']
        return action * std + mean

    def step(self, poses, positions, features, use_features=False, use_positions=False):
        """
        input: image sequence, action sequence
        """
        self.eval()
        N = poses.shape[0]
        action = self(poses, features, positions) 
        return self.denorm_action(action).squeeze(), {}