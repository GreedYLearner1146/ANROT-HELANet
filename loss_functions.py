!pip install pytorch-metric-learning
from pytorch_metric_learning import losses

criterion1 = nn.CrossEntropyLoss()                      # Cross entropy loss as first loss.
optimizer = optim.Adam(model.parameters(), lr=0.001)    # Optimizer with learning rate of 1e-3.


Hesimloss= HesimLoss(temperature=0.01) 

def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )

    loss1 = criterion1(classification_scores, query_labels.cuda())
    loss2 = Hesimloss(classification_scores, query_labels.cuda())
    loss =  loss1 + 0.5*loss2 
    loss.backward()
    optimizer.step()

    return loss.item()
