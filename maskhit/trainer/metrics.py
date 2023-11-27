import pandas as pd
import numpy as np
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt

def c_index(times, scores, events):
    try:
        cindex = concordance_index(times, scores, events)
    except Exception as e:
        cindex = 0.5
    return cindex


def xyear_auc(preds, times, events, cutpoint=5):
    preds = preds.reshape(-1)
    times = times.reshape(-1)
    events = events.reshape(-1)
    ebt = np.zeros_like(times) + 1
    ebt[(times < cutpoint) & (events == 0)] = -1
    ebt[times >= cutpoint] = 0
    ind = ebt >= 0
    try:
        auc = roc_auc_score(ebt[ind], preds[ind])
    except ValueError:
        auc = 0.5
    return auc


def find_confident_instance(preds):
    return preds[preds.max(1).argmax()]

def read_and_adjust_csv(file_name, last_max_id):
    df = pd.read_csv(file_name)

    # Adjust the ids
    df.iloc[:, 1] += last_max_id + 1
    
    # Extract the required columns into numpy arrays
    id_array = df.iloc[:, 1].values.reshape(-1, 1)
    preds_array = df.iloc[:, 2:6].values
    targets_array = df.iloc[:, 6].values.reshape(-1, 1)

    return id_array, preds_array, targets_array

def analyze_predictions():
    print("Analyzing Predictions")

    # List of files
    num_files = 5
    base_path = 'predictions/ibd_project/2023_5_30_new-test-'
    files = [f"{base_path}{i}-predictions.csv" for i in range(num_files)]
    
    # Initialize containers for aggregated data
    agg_ids = np.array([]).reshape(-1, 1)
    agg_preds = np.array([]).reshape(-1, 4)
    agg_targets = np.array([]).reshape(-1, 1)


    # Initialize last maximum id
    last_max_id = -1

    for file in files:
        ids, preds, targets = read_and_adjust_csv(file, last_max_id)
        
        # Update the last_max_id for the next iteration
        last_max_id = np.max(ids)
        
        # Aggregate data
        agg_ids = np.vstack([agg_ids, ids])
        agg_preds = np.vstack([agg_preds, preds])
        agg_targets = np.vstack([agg_targets, targets])
    

    res = calculate_metrics(agg_ids, agg_preds, agg_targets, outcome_type='classification', mode='test')
    print(res)


def calculate_metrics(ids, preds, targets, outcome_type='survival', label_classes = ['Inactive', 'Mild', 'Moderate', 'Severe'], mode = ''):
    if outcome_type == 'survival':
        df = pd.DataFrame(np.concatenate([ids, targets, preds], axis=1))
        df.columns = ['id', 'time', 'event', 'pred']
        df = df.groupby('id').mean()
        c = c_index(df.time, -df.pred, df.event)
        auc_2yr = xyear_auc(df.pred.to_numpy(),
                            df.time.to_numpy(),
                            df.event.to_numpy(),
                            cutpoint=2)
        auc_5yr = xyear_auc(df.pred.to_numpy(),
                            df.time.to_numpy(),
                            df.event.to_numpy(),
                            cutpoint=5)

        res = {'c-index': c, 'auc-2yr': auc_2yr, 'auc-5yr': auc_5yr}
        return res

    elif outcome_type == 'classification':
        df = pd.DataFrame(np.concatenate([ids, targets, softmax(preds, axis=1)], axis=1))
        targets = df.iloc[:, :2].groupby(0).mean().to_numpy().astype(int)
        preds = df.groupby(0).apply(lambda x: find_confident_instance(x.to_numpy()[:, 2:]))
        preds = np.stack(preds.to_list())
        f1 = f1_score(targets, preds.argmax(axis=1), average='weighted')

        if len(np.unique(targets)) != preds.shape[1]:
            auc_score = 0.5
        else:
            try:
                if preds.shape[1] > 2:
                    # multi-class
                    auc_score = roc_auc_score(targets.reshape(-1),
                                        torch.softmax(torch.tensor(preds),
                                                      dim=1),
                                        multi_class='ovr')
                    
                    if mode == 'test' or mode == 'val':
                        # displaying confusion matrix
                        cm = confusion_matrix(targets, preds.argmax(axis = 1))
                        show_confusion_matrix(cm = cm, label_classes = label_classes)

                        # saving multi-class AUC
                        probs = torch.softmax(torch.tensor(preds), dim=1)
                        save_multi_class_auc(probs, targets, label_classes, save_path = 'auc_plot_multi_class.png')

                else:
                    # for binary classification
                    auc_score = roc_auc_score(
                        targets.reshape(-1),
                        torch.softmax(torch.tensor(preds), dim=1)[:, 1])

                    if mode == 'test':
                        # Calculate the FPR and TPR for all thresholds
                        fpr, tpr, _ = roc_curve(targets.reshape(-1), torch.softmax(torch.tensor(preds), dim=1)[:, 1])
                        plot_binary_AUC(fpr, tpr)

            except Exception as e:
                print(e)
                auc_score = 0.5

        res = {'f1': f1, 'auc': auc_score}
        return res

def save_multi_class_auc(probs, targets, label_classes, save_path):
    auc_scores = []

    for i in range(len(label_classes)):
        auc_score = roc_auc_score(targets == i, probs[:, i])
        auc_scores.append(auc_score)
        fpr, tpr, _ = roc_curve(targets == i, probs[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{label_classes[i]} (AUC = {roc_auc:.4f})')
                   
    plt.legend()
    plt.savefig('auc_plot_multi_class.png')
    plt.clf()


def plot_binary_AUC(fpr, tpr):
    # Calculate the AUC (Area under the ROC Curve)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('auc_plot.png')
    plt.clf()  # Clear the current figure

def show_confusion_matrix(cm, label_classes, save_path='confusion_matrix.png'):
    print(f"Length of label classes: {len(label_classes)}")
    df_cm = pd.DataFrame(
        cm, 
        index=label_classes,
        columns=label_classes
    )
    
    # Plot the heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="g", cmap="Blues")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save to a file if save_path is provided
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



class ModelEvaluation(object):

    def __init__(self,
                 outcome_type='survival',
                 loss_function=None,
                 mode='train',
                 variables=['ids', 'preds', 'targets'],
                 device=torch.device('cpu'),
                 timestr=None):

        self.outcome_type = outcome_type
        self.criterion = loss_function
        self.mode = mode
        self.timestr = timestr
        self.variables = variables
        self.device = device
        self.reset()

    def reset(self):
        self.data = {}
        for var in self.variables:
            self.data[var] = None

    def update(self, batch):
        for k, v in batch.items():
            if self.data[k] is None:
                self.data[k] = v.data.cpu().numpy()
            else:
                self.data[k] = np.concatenate(
                    [self.data[k], v.data.cpu().numpy()])

    def evaluate(self, mode = ''):
        metrics = calculate_metrics(self.data['ids'],
                                    self.data['preds'],
                                    self.data['targets'],
                                    outcome_type=self.outcome_type, mode = mode)

        loss_epoch = self.criterion.calculate(
            torch.tensor(self.data['preds']).to(self.device),
            torch.tensor(self.data['targets']).to(self.device))

        metrics['loss'] = loss_epoch.item()

        return metrics

    def save(self, filename):
        values = []
        for k, v in self.data.items():
            values.append(v)
        df = pd.DataFrame(np.concatenate(values, 1))
        if filename is None:
            return df
        else:
            df.to_csv(filename)
