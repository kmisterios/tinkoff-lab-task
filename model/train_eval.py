from model.model import MultiTaskBert
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm import trange
from torch.nn import BCEWithLogitsLoss
import torch
from transformers import  AdamW
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def evaluate(eval_dataloader, 
             pretrained_model,
             method_name, 
             checkpoints_path, 
             config, 
             metric_funcs = {"cola" : matthews_corrcoef,"sst2" : accuracy_score,"rte" : accuracy_score},
             finetune_model = MultiTaskBert):

    """Evaluates the trained model. Weights are loaded from checkpoint.

        Args:
            eval_dataloader: evaluation singletask dataloader
            pretrained_model: pretrained model
            method_name: name of method to evaluate. One of ``SingleHeadMultitask``, ``MultiHeadMultitask``, 
            ``cola_Singletask``, ``sst2_Singletask``, ``rte_Singletask``.
                * ``SingleHeadMultitask`` - method with shared encoder and 1 classifier for all tasks.
                * ``MultiHeadMultitask`` - method with shared encoder and seperate classifier for each task.
                * ``cola_Singletask``, ``sst2_Singletask``, ``rte_Singletask`` - singletask models with seperate 
                model for each task
            checkpoints_path: path for checkpoints folder
            config: configuration dictionary for chosen method
            metric_funcs: dictionary of sklearn metric functions for the tasks. Default: ``matthews_corrcoef`` for
            'cola' task and ``accuracy_score`` for other tasks
            finetune_model: fine-tuning model. Default: ``MultiTaskBert`` provided in this repository
    """
    
    classifier = finetune_model(pretrained_model, config["linear_layer_size"], task_names=config["task_names"])
    classifier.load_state_dict(torch.load(f"{checkpoints_path}/model_best_{method_name}_{config['num_epochs']}.pth"))
    classifier.to(DEVICE)
    
    classifier.eval()
    all_preds, all_labels = [], []
    for (batch, task_name) in eval_dataloader:
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            task_name_ = "all" if config["mode"] == "singlehead" else task_name
            logits = classifier(input_ids, attention_mask, task_name_)
            
        preds = logits.argmax(dim = 1).cpu()
        labels = labels.argmax(dim = 1).cpu()
        all_preds += list(preds)
        all_labels += list(labels)


    score = metric_funcs[task_name](all_labels, all_preds)
    print(f"Best {metric_funcs[task_name].__name__} for {method_name} for {task_name}: {score}")
    return score


def train(train_dataloader,
          valid_dataloader,
          pretrained_model, 
          config,
          method_name,
          checkpoints_path = "./checkpoints",
          metric_funcs = {"cola" : matthews_corrcoef,"sst2" : accuracy_score,"rte" : accuracy_score},
          draw_graphs = True,
          finetune_model = MultiTaskBert
         ):

    """Trains the model, evaluating the performance each epoch and creating checkpoints.

        Args:
            train_dataloader: training multitask dataloader
            valid_dataloader: validation multitask dataloader
            pretrained_model: pretrained model
            config: configuration dictionary for chosen method
            method_name: name of method to evaluate. One of ``SingleHeadMultitask``, ``MultiHeadMultitask``, 
            ``cola_Singletask``, ``sst2_Singletask``, ``rte_Singletask``.
                * ``SingleHeadMultitask`` - method with shared encoder and 1 classifier for all tasks.
                * ``MultiHeadMultitask`` - method with shared encoder and seperate classifier for each task.
                * ``cola_Singletask``, ``sst2_Singletask``, ``rte_Singletask`` - singletask models with seperate 
                model for each task
            checkpoints_path: path for checkpoints folder
            metric_funcs: dictionary of sklearn metric functions for the tasks. Default: ``matthews_corrcoef`` for
            'cola' task and ``accuracy_score`` for other tasks
            draw_graphs: If ``True``, the loss and metrics evolution will be drawn each epoch. Default: ``True``
            finetune_model: fine-tuning model. Default: ``MultiTaskBert`` provided in this repository
    """

    classifier = finetune_model(pretrained_model, config["linear_layer_size"], task_names=config["task_names"])
    optimizer = AdamW(classifier.parameters(), lr=2e-5, eps = 1e-08)
    scheduler = LinearLR(optimizer, total_iters=3)
    criterion = BCEWithLogitsLoss()
    num_epochs = config["num_epochs"]

    torch.cuda.empty_cache()

    classifier.to(DEVICE)
    max_grad_norm = 1.0
    loss_values, validation_loss_values = [], []
    avg_score = []
    prev_metric = -np.inf
    try:
        for ep in trange(num_epochs, desc="Epoch"):
            classifier.train()
            total_loss = 0

            for step, (batch, task_name) in enumerate(train_dataloader):

                batch = tuple(t.to(DEVICE) for t in batch)

                input_ids, attention_mask, labels  = batch

                optimizer.zero_grad()
                task_name_ = "all" if config["mode"] == "singlehead" else task_name
                logits = classifier(input_ids, attention_mask, task_name_)
                loss = criterion(logits, labels)
                loss.backward()
                #normalize loss with the size of dataset to have the loss component of the same scale
                new_loss = loss.item() / train_dataloader.lengths[task_name] if config["mode"] != "singletask" else loss.item()
                total_loss += new_loss
                torch.nn.utils.clip_grad_norm_(parameters=classifier.parameters(), max_norm=max_grad_norm)

                optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)
            clear_output(wait=True)
            print("\nAverage train loss: {}".format(avg_train_loss))

            loss_values.append(avg_train_loss)


            classifier.eval()
            eval_loss = 0
            all_preds = {label: [] for label in valid_dataloader.task_names}
            all_labels = {label: [] for label in valid_dataloader.task_names}
            for (batch, task_name) in valid_dataloader:
                batch = tuple(t.to(DEVICE) for t in batch)
                input_ids, attention_mask, labels = batch

                task_name_ = "all" if config["mode"] == "singlehead" else task_name
                with torch.no_grad():
                    logits = classifier(input_ids, attention_mask, task_name_)
                
                loss = criterion(logits, labels)
                new_loss = loss.item() / valid_dataloader.lengths[task_name] if config["mode"] != "singletask" else loss.item()
                eval_loss += new_loss
                preds = logits.argmax(dim = 1).cpu()
                labels = labels.argmax(dim = 1).cpu()
                all_preds[task_name] += list(preds)
                all_labels[task_name] += list(labels)

            eval_loss = eval_loss / len(valid_dataloader)
            scheduler.step()

            validation_loss_values.append(eval_loss)
            print("Validation loss: {}".format(eval_loss))
            scores = []
            tasks_list = valid_dataloader.task_names if config["mode"] != "singletask" else config["task_names"]
            for task_name in tasks_list:
                single_metric = metric_funcs[task_name](all_labels[task_name], all_preds[task_name])
                scores.append(single_metric)
                print(f"Validation {metric_funcs[task_name].__name__} for {task_name} task: {single_metric}")

            metric = sum(scores) / len(scores)
            avg_score.append(metric)
            print(f"Average score: {metric}")
            if metric > prev_metric:
                torch.save(classifier.state_dict(), f"{checkpoints_path}/model_best_{method_name}_{num_epochs}.pth")
                print(f"Best model saved at epoch {ep}")
                prev_metric = metric

            print("Best score: ", prev_metric)
            torch.save(classifier.state_dict(), f"{checkpoints_path}/model_curr_{method_name}_{num_epochs}.pth")

            if draw_graphs:
                plt.figure(figsize=(10,8))
                plt.plot(loss_values, label="Train loss")
                plt.plot(validation_loss_values, label="Validation loss")
                plt.xlabel("#epoch")
                plt.ylabel("Loss value")
                plt.grid()
                plt.legend()
                plt.show()

                plt.figure(figsize=(10,8))
                plt.plot(avg_score, label="score", c="orange")
                plt.xlabel("#epoch")
                plt.ylabel("Average validation score")
                plt.grid()
                plt.legend()
                plt.show()
    except KeyboardInterrupt:
        pass