import torch.nn as nn

class MultiTaskBert(nn.Module):
    def __init__(self, pretrained_model, linear_layer_size, task_names = ["all"]):
        super(MultiTaskBert, self).__init__()
        self.model = pretrained_model
        #to have multiple heads or only one
        self.fc = nn.ModuleDict({ key : nn.Linear(linear_layer_size, 2, bias=True) for key in task_names })

    def forward(self, input_ids, attention_mask, task_name = "all"):
        #we take only embedding of [CLS] token, it's first in the output layer
        x = self.model(input_ids, attention_mask)["last_hidden_state"][:,0]
        logits = self.fc[task_name](x)
        return logits