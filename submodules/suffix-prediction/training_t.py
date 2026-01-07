import torch
import torchmetrics
from evaluation import evaluate_accuracy_next_activity, logic_loss, logic_loss_multiple_samples
import sys
if torch.cuda.is_available():
     device = 'cuda:0'
else:
    device = 'cpu'
from statistics import mean

def train(model, train_dataset, test_dataset, max_num_epochs, epsilon, deepdfa=None, prefix_len=0, batch_size=32, logic_loss_type="one_sample"):
    curr_temp = 0.5
    lambda_temp = 0.9999999999
    min_temp = 0.0001
    
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    
    # Lower learning rate for Transformer models
    is_transformer = 'Transformer' in model.__class__.__name__
    lr = 0.0001 if is_transformer else 0.0005
    
    # Add weight decay for regularization with Transformer
    weight_decay = 0.01 if is_transformer else 0
    
    # Use AdamW for Transformer (better handles weight decay)
    if is_transformer:
        optim = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    # Initialize accuracy metric
    acc_func = torchmetrics.Accuracy(task="multiclass", num_classes=train_dataset.size()[-1], top_k=1).to(device)
    
    old_loss = 1000
    
    # For Transformer learning rate scheduling
    if is_transformer:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_num_epochs)
        warmup_epochs = 10
        warmup_factor = 0.1
    
    ############################ TRAINING
    for epoch in range(max_num_epochs*2):
        model.train()
        current_index = 0
        train_acc_batches = []
        sup_loss_batches = []
        log_loss_batches = []
        
        # Learning rate warmup for Transformer
        if is_transformer and epoch < warmup_epochs:
            for param_group in optim.param_groups:
                param_group['lr'] = lr * (warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs))
        
        while current_index < train_dataset.size()[0]:
            initial = current_index
            final = min(current_index + batch_size, train_dataset.size()[0])
            current_index = final
            
            # Prepare batch data
            X = train_dataset[initial:final, :-1, :].to(device)
            Y = train_dataset[initial:final, 1:, :]
            target = torch.argmax(Y.reshape(-1, Y.size()[-1]), dim=-1).to(device)
            
            optim.zero_grad()
            
            #################################### supervised loss
            predictions, _ = model(X)
            predictions = predictions.reshape(-1, predictions.size()[-1])
            
            sup_loss = loss_func(predictions, target)
            sup_loss_batches.append(sup_loss.item())
            
            ##################################### logic loss
            # Implementation for multiple_samples logic loss
            if logic_loss_type == "multiple_samples" and deepdfa is not None:
                log_loss = logic_loss_multiple_samples(model, deepdfa, X, prefix_len, curr_temp, num_samples=10)
                log_loss_batches.append(log_loss.item())
                loss = 0.6*sup_loss + 0.4*log_loss
            # Implementation for one_sample logic loss after warm-up
            elif epoch > 500 and deepdfa is not None:
                if logic_loss_type == "one_sample":
                    log_loss = logic_loss(model, deepdfa, X, prefix_len, curr_temp)
                    log_loss_batches.append(log_loss.item())
                    loss = 0.6 * sup_loss + 0.4 * log_loss
                else:
                    loss = sup_loss
            else:
                loss = sup_loss
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping (especially helpful for Transformer)
            if is_transformer:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optim.step()
            
            # Track accuracy
            train_acc_batches.append(acc_func(predictions, target).item())
        
        # Update learning rate scheduler for Transformer
        if is_transformer and epoch >= warmup_epochs:
            scheduler.step()
        
        # Calculate statistics
        train_acc = mean(train_acc_batches)
        sup_loss = mean(sup_loss_batches)
        
        # Check temperature
        if curr_temp == min_temp:
            print("MINIMUM TEMPERATURE REACHED")
        
        # Evaluate on test set
        model.eval()
        test_acc = evaluate_accuracy_next_activity(model, test_dataset, acc_func)
        
        # Logging
        if epoch % 100 == 0:
            if deepdfa is None or (logic_loss_type == "one_sample" and epoch <= 500):
                print(f"Epoch {epoch}:\tloss:{sup_loss:.6f}\ttrain accuracy:{train_acc:.4f}\ttest accuracy:{test_acc:.4f}")
                loss = sup_loss
            else:
                log_loss = mean(log_loss_batches)
                print(f"Epoch {epoch}:\tloss:{sup_loss:.6f}\tlogic_loss:{log_loss:.6f}\ttrain accuracy:{train_acc:.4f}\ttest accuracy:{test_acc:.4f}")
                loss = 0.6*sup_loss + 0.4*log_loss
            
            # For Transformers, print learning rate
            if is_transformer:
                print(f"Learning rate: {optim.param_groups[0]['lr']:.6f}")
        
        # Early stopping based on loss
        if loss < epsilon:
            return train_acc, test_acc
        
        if epoch > 500 and abs(loss - old_loss) < 0.00001:
            return train_acc, test_acc
            
        old_loss = loss
    
    return train_acc, test_acc