#%%
import torch
import torch.nn as nn
from tqdm import tqdm
from train_utils import OptimizationConfig, TrainingConfig, LoggingConfig, OnlineMovingAverage, ema_avg_fn, move_to_device
from torch.optim.swa_utils import AveragedModel
import os
import sys
import matplotlib
matplotlib.use("Agg")

# %%
def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    logger: LoggingConfig,
):

    model = model.to(config.device)
    state = logger.load_latest_checkpoint()
    if state is not None:
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state['scheduler_state_dict'])
        global_step = state["global_step"]
        start_epoch = state["epoch"]
    else:
        global_step = 0
        start_epoch = 0
    
    logger.global_step = global_step
    train_avg_loss = OnlineMovingAverage(size=5000)
    test_avg_loss = OnlineMovingAverage(size=1000)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, config.num_epochs):
        pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", mininterval=10)
        for input_ids, attention_mask, labels in pb:
            model.train()
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            pred_labels,_,_ = model(input_ids, mask=attention_mask)
            loss = criterion(pred_labels, labels)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            train_avg_loss.update(loss.item()/len(input_ids))
            pb.set_description(f"Avg_training_loss: {train_avg_loss.mean:.3e}")
            
            if ((logger.global_step + 1) % logger.log_loss_freq == 0) or (logger.global_step == 0):
                
                with torch.no_grad():
                    input_ids_test, attention_mask_test, labels_test =next(iter(val_loader))
                    input_ids_test = input_ids_test.to(config.device) 
                    attention_mask_test = attention_mask_test.to(config.device)
                    labels_test = labels_test.to(config.device)

                    pred_labels_test,_,_ = model(input_ids_test, mask=attention_mask_test)
                    test_loss = criterion(pred_labels_test, labels_test).item()
                
                test_avg_loss.update(test_loss/len(input_ids_test))
                del input_ids_test, attention_mask_test, labels_test, pred_labels_test, test_loss
                torch.cuda.empty_cache()    

                metrics = {
                    "val_loss": test_avg_loss.mean,
                    "train_loss": train_avg_loss.mean,
                    "lr": optimizer.param_groups[0]["lr"],
                    "max_grad_norm": grad_norm.max()
                }

                logger.log_metrics(metrics, logger.global_step)
                logger.log_histogram(grad_norm, "grad_norm", logger.global_step)

            logger.global_step += 1

        # Save checkpoint
        if ((epoch + 1) == config.num_epochs) or (epoch % logger.save_freq == 0):
            state = {
                "model_state_dict": model.state_dict(),
                "global_step": logger.global_step,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None
            }

            logger.save_checkpoint(state, epoch, metric_value=test_avg_loss.mean)
            if  epoch % logger.save_freq == 0:
                logger.clean_old_checkpoint()
                
    # final evaluation
    test_corrects = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    
    print(f"Accuracy evaluation on test set: {(test_corrects/total):.2f}")

                

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from models import BertClf
    from transformers import  DistilBertForSequenceClassification
    from utils import NlpDataset
    import argparse  
    import pandas as pd
    from transformers import DistilBertTokenizerFast
    import torch.utils.data as data


    parser = argparse.ArgumentParser(description="Training script for movie plot prediction classification task.")
    parser.add_argument("--num_epochs", type=int, default=300, help="Numeber of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--save_dir", type=str, help="Saved directory",
                        default='exp/default')
    args = parser.parse_args()

    #################### DEFINE DATASET ##############################
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    with open(os.path.join(project_path, "content/movie_plot/movie_categories.txt"), "r") as f:
        movie_categories = [line.strip() for line in f.readlines()]

    cat2id = {cat: idx for idx, cat in enumerate(movie_categories)}

    train_df = pd.read_csv(os.path.join(project_path, "content/movie_plot/train_movie_plots.csv"))
    train_plot = train_df["movie_plot"]
    train_labels = train_df["movie_category"].map(cat2id)

    test_df = pd.read_csv(os.path.join(project_path, "content/movie_plot/test_movie_plots.csv"))
    test_plot = test_df["movie_plot"]
    test_labels = test_df["movie_category"].map(cat2id)

    train_size = len(train_df)
    test_size = len(test_df)
    print(f"Train set size: {train_size}, test set size: {test_size}")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = NlpDataset(train_plot, train_labels, tokenizer)
    test_dataset = NlpDataset(test_plot,test_labels, tokenizer)


    if train_size < args.batch_size:
        sampler = data.RandomSampler(train_dataset, replacement=True, num_samples=args.batch_size)
        shuffle = False
    else:
        sampler = None
        shuffle = True


    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=shuffle, pin_memory=True, sampler=sampler,
                                   drop_last=False, num_workers=8)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    ########################### DEFINE MODEL ########################
    classes = movie_categories
    num_classes = len(classes)
    
    distilbert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                  num_labels=num_classes,
                                                                  output_attentions=True,
                                                                  output_hidden_states=True)
    model = BertClf(distilbert)
    

    ######################### Training Config and training logger ##################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Create training configuration
    training_config = TrainingConfig()
    training_config.update(**vars(args))

    # Create logging configuration
    logger = LoggingConfig(project_dir=os.path.join(project_path, args.save_dir),
                           exp_name=f"bertcls_{train_size}")
    logger.monitor_metric = "test_avg_loss"
    logger.monitor_mode = "min"
    logger.num_log_images = 5
    logger.initialize()
    logger.log_hyperparameters(vars(args), main_key="training_config")


    # Save checkpoint every 400 steps
    num_step_per_epoch = max(len(train_loader), 1)
    freq = max(1, int(400 // num_step_per_epoch))
    logger.save_freq = freq
    logger.val_epoch_freq = freq
    logger.log_loss_freq = 5

    optim_config = OptimizationConfig()
    optimizer = optim_config.get_optimizer(model)
    lr_scheduler = optim_config.get_scheduler(optimizer)


    ########################## Lance training loop ###############################
    training_loop(model=model,
                  optimizer=optimizer,
                  lr_scheduler=lr_scheduler,
                  train_loader=train_loader,
                  val_loader=test_loader,
                  config=training_config,
                  logger=logger)




# %%
