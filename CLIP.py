import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, TextEncoder, ProjectionHead
from coformer.contrCoformer import contrCoformer, d_model


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])  # shape: (batch_size, embedding_dim)
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]  # shape: (batch_size, max_length, embedding_dim)
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)  # shape: (batch_size, projection_dim)
        text_embeddings = self.text_projection(text_features)  # shape: (batch_size, projection_dim)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature  # shape: (batch_size, batch_size)
        images_similarity = image_embeddings @ image_embeddings.T  # shape: (batch_size, batch_size)
        texts_similarity = text_embeddings @ text_embeddings.T  # shape: (batch_size, batch_size)
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )   # shape: (batch_size, batch_size)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()



class constrastiveTraj(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        embedding_dim=d_model,
    ):
        super().__init__()
        self.temperature = temperature
        self.first_encoder = contrCoformer()
        self.second_encoder = contrCoformer()
        self.first_projection = ProjectionHead(embedding_dim=embedding_dim)
        self.second_projection = ProjectionHead(embedding_dim=embedding_dim)

    def forward(self, batch):
        # Getting Image and Text Features
        first_features = self.first_encoder.encode(**batch["traj_1"])
        second_features = self.second_encoder.encode(**batch["traj_2"])
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.first_projection(first_features)
        text_embeddings = self.second_projection(second_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

if __name__ == '__main__':
    # images = torch.randn(8, 3, 224, 224)
    # input_ids = torch.randint(5, 300, size=(8, 25))
    # attention_mask = torch.ones(8, 25)
    # batch = {
    #     'image': images,
    #     'input_ids': input_ids,
    #     'attention_mask': attention_mask
    # }

    # CLIP = CLIPModel()
    # loss = CLIP(batch)
    # print("")

    poses_1 = torch.randn(4, 100, 7)
    features_1 = torch.randn(4, 5, 6528)
    positions_1 = torch.randn(4, 100, 5, 2)
    traj_1 = {'poses': poses_1, 'features': features_1, 'positions': positions_1}

    poses_2 = torch.randn(4, 100, 7)
    features_2 = torch.randn(4, 5, 6528)
    positions_2 = torch.randn(4, 100, 5, 2)
    traj_2 = {'poses': poses_2, 'features': features_2, 'positions': positions_2}

    batch = {'traj_1': traj_1, 'traj_2': traj_2}
    contrTraj = constrastiveTraj()
    loss = contrTraj(batch)