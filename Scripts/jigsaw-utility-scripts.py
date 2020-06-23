import os, random, time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

    
# Auxiliary functions
def color_map(val):
    if type(val) == float:
        if val <= 0.2:
            color = 'red'
        elif val <= 0.3:
            color = 'orange'
        elif val >= 0.8:
            color = 'green'
        else:
            color = 'black'
    else:
        color = 'black'
    return 'color: %s' % color

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
def set_up_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    return strategy, tpu
    
    
# Model evaluation    
def evaluate_model(k_fold, n_folds=1, label_col='toxic'):
    metrics_df = pd.DataFrame([], columns=['Metric', 'Train', 'Valid', 'Var'])
    metrics_df['Metric'] = ['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Support']
    
    for n_fold in range(n_folds):
        rows = []
        train_set = k_fold[k_fold['fold_%d' % (n_fold+1)] == 'train']
        validation_set = k_fold[k_fold['fold_%d' % (n_fold+1)] == 'validation'] 
        
        train_report = classification_report(train_set[label_col], train_set['pred_%d' % (n_fold+1)], output_dict=True)
        valid_report = classification_report(validation_set[label_col], validation_set['pred_%d' % (n_fold+1)], output_dict=True)
    
        rows.append([roc_auc_score(train_set[label_col], train_set['pred_%d' % (n_fold+1)]),
                     roc_auc_score(validation_set[label_col], validation_set['pred_%d' % (n_fold+1)])])
        rows.append([train_report['accuracy'], valid_report['accuracy']])
        rows.append([train_report['1']['precision'], valid_report['1']['precision']])
        rows.append([train_report['1']['recall'], valid_report['1']['recall']])
        rows.append([train_report['1']['f1-score'], valid_report['1']['f1-score']])
        rows.append([train_report['1']['support'], valid_report['1']['support']])
        
        metrics_df = pd.concat([metrics_df, pd.DataFrame(rows, columns=['Train_fold_%d' % (n_fold+1), 
                                                                        'Valid_fold_%d' % (n_fold+1)])], axis=1)
    
    metrics_df['Train'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Train_fold')]].mean(axis=1)
    metrics_df['Valid'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Valid_fold')]].mean(axis=1)
    metrics_df['Var'] = metrics_df['Train'] - metrics_df['Valid']
    
    return metrics_df.set_index('Metric')

def evaluate_model_single_fold(k_fold, n_fold=1, label_col='toxic'):
    metrics_df = pd.DataFrame([], columns=['Metric', 'Train', 'Valid', 'Var'])
    metrics_df['Metric'] = ['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Support']
    
    rows = []
    fold_col = f'fold_{n_fold}' 
    pred_col = f'pred_{n_fold}' 
    train_set = k_fold[k_fold[fold_col] == 'train']
    validation_set = k_fold[k_fold[fold_col] == 'validation'] 

    train_report = classification_report(train_set[label_col], train_set[pred_col], output_dict=True)
    valid_report = classification_report(validation_set[label_col], validation_set[pred_col], output_dict=True)

    rows.append([roc_auc_score(train_set[label_col], train_set[pred_col]),
                 roc_auc_score(validation_set[label_col], validation_set[pred_col])])
    rows.append([train_report['accuracy'], valid_report['accuracy']])
    rows.append([train_report['1']['precision'], valid_report['1']['precision']])
    rows.append([train_report['1']['recall'], valid_report['1']['recall']])
    rows.append([train_report['1']['f1-score'], valid_report['1']['f1-score']])
    rows.append([train_report['1']['support'], valid_report['1']['support']])

    metrics_df = pd.concat([metrics_df, pd.DataFrame(rows, columns=['Train_' + fold_col, 
                                                                    'Valid_' + fold_col])], axis=1)
    
    metrics_df['Train'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Train_fold')]].mean(axis=1)
    metrics_df['Valid'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Valid_fold')]].mean(axis=1)
    metrics_df['Var'] = metrics_df['Train'] - metrics_df['Valid']
    
    return metrics_df.set_index('Metric')

def evaluate_model_lang(df, n_folds, label_col='toxic', pred_col='pred'):
    metrics_df = pd.DataFrame([], columns=['Lang / ROC AUC', 'Mean'])
    metrics_df['Lang / ROC AUC'] = ['Overall'] + [l for l in df['lang'].unique()]
    
    for n_fold in range(n_folds):
        rows = []
        rows.append([roc_auc_score(df[label_col], df[pred_col + '_%d' % (n_fold+1)])])
        
        for lang in df['lang'].unique():
            subset = df[df['lang'] == lang]
            rows.append([roc_auc_score(subset[label_col], subset[pred_col + '_%d' % (n_fold+1)])])

        metrics_df = pd.concat([metrics_df, pd.DataFrame(rows, columns=['Fold_%d' % (n_fold+1)])], axis=1)
    
    metrics_df['Mean'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Fold')]].mean(axis=1)
    
    return metrics_df.set_index('Lang / ROC AUC')

def evaluate_model_single_fold_lang(df, n_fold, label_col='toxic', pred_col='pred'):
    metrics_df = pd.DataFrame([], columns=['Lang / ROC AUC', 'Mean'])
    metrics_df['Lang / ROC AUC'] = ['Overall'] + [l for l in df['lang'].unique()]
    
    rows = []
    fold_col = f'Fold_{n_fold}' 
    rows.append([roc_auc_score(df[label_col], df[f'{pred_col}_{n_fold}'])])

    for lang in df['lang'].unique():
        subset = df[df['lang'] == lang]
        rows.append([roc_auc_score(subset[label_col], subset[f'{pred_col}_{n_fold}'])])

    metrics_df = pd.concat([metrics_df, pd.DataFrame(rows, columns=[fold_col])], axis=1)
    
    metrics_df['Mean'] = metrics_df[[c for c in metrics_df.columns if c.startswith('Fold')]].mean(axis=1)
    
    return metrics_df.set_index('Lang / ROC AUC')

def plot_metrics(history):
    metric_list = list(history.keys())
    size = len(metric_list)//2
    fig, axes = plt.subplots(size, 1, sharex='col', figsize=(20, size * 5))
    axes = axes.flatten()
    
    for index in range(len(metric_list)//2):
        metric_name = metric_list[index]
        val_metric_name = metric_list[index+size]
        axes[index].plot(history[metric_name], label='Train %s' % metric_name)
        axes[index].plot(history[val_metric_name], label='Validation %s' % metric_name)
        axes[index].legend(loc='best', fontsize=16)
        axes[index].set_title(metric_name)

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.show()
    
def plot_metrics_2(history):
    metric_list = list(history.keys())
    size = len(metric_list)//3
    fig, axes = plt.subplots(size, 1, sharex='col', figsize=(20, size * 5))
    axes = axes.flatten()
    
    for index in range(len(metric_list)//3):
        metric_name = metric_list[index]
        val_metric_name = metric_list[index+(size*1)]
        val_2_metric_name = metric_list[index+(size*2)]
        axes[index].plot(history[metric_name], label='Train %s' % metric_name)
        axes[index].plot(history[val_metric_name], label='Validation %s' % metric_name)
        axes[index].plot(history[val_2_metric_name], label='Validation_2 %s' % metric_name)
        axes[index].legend(loc='best', fontsize=16)
        axes[index].set_title(metric_name)

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.show()
    
def plot_confusion_matrix(y_train, train_pred, y_valid, valid_pred, labels=[0, 1]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    train_cnf_matrix = confusion_matrix(y_train, train_pred)
    validation_cnf_matrix = confusion_matrix(y_valid, valid_pred)

    train_cnf_matrix_norm = train_cnf_matrix.astype('float') / train_cnf_matrix.sum(axis=1)[:, np.newaxis]
    validation_cnf_matrix_norm = validation_cnf_matrix.astype('float') / validation_cnf_matrix.sum(axis=1)[:, np.newaxis]

    train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=labels, columns=labels)
    validation_df_cm = pd.DataFrame(validation_cnf_matrix_norm, index=labels, columns=labels)

    sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues",ax=ax1).set_title('Train')
    sns.heatmap(validation_df_cm, annot=True, fmt='.2f', cmap=sns.cubehelix_palette(8),ax=ax2).set_title('Validation')
    plt.show()
    
    
# Datasets
def get_training_dataset(x_train, y_train, batch_size, buffer_size, seed=0):
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': x_train[0], 
                                                   'attention_mask': x_train[1]}, y_train))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048, seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size)
    return dataset

def get_validation_dataset(x_valid, y_valid, batch_size, buffer_size, repeated=False, seed=0):
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': x_valid[0], 
                                                   'attention_mask': x_valid[1]}, y_valid))
    if repeated:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048, seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size)
    return dataset

def get_test_dataset(x_test, batch_size, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices({'input_ids': x_test[0], 
                                                  'attention_mask': x_test[1]})
    dataset = dataset.batch(batch_size)
    return dataset


# Training
def custom_fit(model, metrics_dict, train_step_fn, valid_step_fn, train, validation, 
               train_step_size, validation_step_size, batch_size, epochs, patience=None, 
               model_path='model.h5', save_last=False, checkpoint_freq=None):
        
    # ==================== Setup training loop ====================
    step = 0
    epoch = 0
    epoch_steps = 0
    epoch_start_time = time.time()
    patience_cnt = 0
    best_val = float("inf")
    
    history = {}
    for metric in metrics_dict.keys():
        history[metric] = []

    print(f'Train for {train_step_size} steps, validate for {validation_step_size} steps')
    # ==================== Train model ====================
    while True:
        train_step_fn(train)
        epoch_steps += train_step_size
        step += train_step_size

        # validation run at the end of each epoch
        if (step // train_step_size) > epoch:
            # validation run
            valid_epoch_steps = 0
            valid_step_fn(validation)
            valid_epoch_steps += validation_step_size
            
            # compute metrics
            for metric in metrics_dict.keys():
                if 'loss' in metric:
                    if 'val_' in metric: # loss from validation
                        history[metric].append(metrics_dict[metric].result().numpy() / (batch_size * valid_epoch_steps))
                    else: # loss from training
                        history[metric].append(metrics_dict[metric].result().numpy() / (batch_size * epoch_steps))
                else: # any other metric
                    history[metric].append(metrics_dict[metric].result().numpy())

            # report metrics
            epoch_time = time.time() - epoch_start_time
            print('\nEPOCH {:d}/{:d}'.format(epoch+1, epochs))
            report = f"time: {epoch_time:0.1f}s"
            for metric in metrics_dict.keys():
                report += f" {metric}: {history[metric][-1]:0.4f}"
            print(report)

            # set up next epoch
            epoch = step // train_step_size
            epoch_steps = 0
            epoch_start_time = time.time()
            for metric in metrics_dict.values():
                metric.reset_states()
                
            # Model checkpoint
            if checkpoint_freq is not None:
                if epoch % checkpoint_freq == 0:
                    model_path_chk = 'ep_%d_%d' % (epoch, model_path)
                    model.save_weights(model_path_chk)
                    print('Checkpointing model weights at "%s"' % model_path_chk)

            if epoch <= epochs:
                if patience is not None:
                    # Early stopping monitor
                    if history['val_loss'][-1] <= best_val:
                        best_val = history['val_loss'][-1]
                        model.save_weights(model_path)
                        print('Saved model weights at "%s"' % model_path)
                    else:
                        patience_cnt += 1
                    if patience_cnt >= patience:
                        print('Epoch %05d: early stopping' % epoch)
                        if epoch < epochs:
                            return history
            if epoch >= epochs:
                if save_last:
                    model_path_last = 'last_' + model_path
                    model.save_weights(model_path_last)
                    print('Training finished saved model weights at "%s"' % model_path_last)
                else:
                    print('Training finished')
                    
                return history
            

def custom_fit_2(model, metrics_dict, train_step_fn, valid_step_fn, valid_2_step_fn, train, validation, validation_2, 
                train_step_size, validation_step_size, validation_2_step_size, batch_size, epochs, patience=None, 
                model_path='model.h5', save_last=False, checkpoint_freq=None):
        
    # ==================== Setup training loop ====================
    step = 0
    epoch = 0
    epoch_steps = 0
    epoch_start_time = time.time()
    patience_cnt = 0
    best_val = float("inf")
    
    history = {}
    for metric in metrics_dict.keys():
        history[metric] = []

    print(f'Train for {train_step_size} steps, validate for {validation_step_size} steps, validate_2 for {validation_2_step_size} steps')
    # ==================== Train model ====================
    while True:
        train_step_fn(train)
        epoch_steps += train_step_size
        step += train_step_size
        
        # validation run at the end of each epoch
        if (step // train_step_size) > epoch:
            # validation run
            valid_epoch_steps = 0
            valid_step_fn(validation)
            valid_epoch_steps += validation_step_size
            # validation_2 run
            valid_2_epoch_steps = 0
            valid_2_step_fn(validation_2)
            valid_2_epoch_steps += validation_2_step_size
            
            # compute metrics
            for metric in metrics_dict.keys():
                if 'loss' in metric:
                    if 'val_loss' in metric: # loss from validation
                        history[metric].append(metrics_dict[metric].result().numpy() / (batch_size * valid_epoch_steps))
                    elif 'val_2_loss' in metric: # loss from validation_2
                        history[metric].append(metrics_dict[metric].result().numpy() / (batch_size * valid_2_epoch_steps))
                    else: # loss from training
                        history[metric].append(metrics_dict[metric].result().numpy() / (batch_size * epoch_steps))
                else: # any other metric
                    history[metric].append(metrics_dict[metric].result().numpy())

            # report metrics
            epoch_time = time.time() - epoch_start_time
            print('\nEPOCH {:d}/{:d}'.format(epoch+1, epochs))
            report = f"time: {epoch_time:0.1f}s"
            for metric in metrics_dict.keys():
                report += f" {metric}: {history[metric][-1]:0.4f}"
            print(report)

            # set up next epoch
            epoch = step // train_step_size
            epoch_steps = 0
            epoch_start_time = time.time()
            for metric in metrics_dict.values():
                metric.reset_states()
                
            # Model checkpoint
            if checkpoint_freq is not None:
                if epoch % checkpoint_freq == 0:
                    model_path_chk = 'ep_%d_%d' % (epoch, model_path)
                    model.save_weights(model_path_chk)
                    print('Checkpointing model weights at "%s"' % model_path_chk)

            if epoch <= epochs:
                if patience is not None:
                    # Early stopping monitor
                    if history['val_loss'][-1] <= best_val:
                        best_val = history['val_loss'][-1]
                        model.save_weights(model_path)
                        print('Saved model weights at "%s"' % model_path)
                    else:
                        patience_cnt += 1
                    if patience_cnt >= patience:
                        print('Epoch %05d: early stopping' % epoch)
                        if epoch < epochs:
                            return history
            if epoch >= epochs:
                if save_last:
                    model_path_last = 'last_' + model_path
                    model.save_weights(model_path_last)
                    print('Training finished saved model weights at "%s"' % model_path_last)
                else:
                    print('Training finished')
                    
                return history