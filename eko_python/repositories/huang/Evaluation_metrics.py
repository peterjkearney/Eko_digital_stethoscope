import os
import numpy as np
import Config
import pickle
from sklearn.model_selection import StratifiedKFold
import shutil
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, accuracy_score
import sklearn.metrics as metrics



def Data_Acq(fileName):
    with open(fileName, 'rb') as f:
        sample = pickle.load(f)
    return sample


def CrossValidation(dir, savedir, cross_subject_flag=True):
    dataDir = os.listdir(dir)

    # Cross_subject
    subjectList = []

    if cross_subject_flag:
        for tempDir in dataDir:
            tempDir = tempDir[:3]
            if tempDir not in subjectList:
                subjectList.append(tempDir)
    else:
        for tempDir in dataDir:
            if tempDir not in subjectList:
                subjectList.append(tempDir)

    # 交叉验证
    seed = 2
    np.random.seed(seed)
    num_k = Config.num_k
    kfold = StratifiedKFold(n_splits=num_k, shuffle=True, random_state=seed)

    for count, (trainIndex, testIndex) in enumerate(kfold.split(subjectList, np.zeros((len(subjectList),)))):
        print(count, "Start")
        if not os.path.exists(f"{savedir}/Fold_{count}"):
            os.makedirs(f"{savedir}/Fold_{count}")
        else:
            shutil.rmtree(f"{savedir}/Fold_{count}")
            os.makedirs(f"{savedir}/Fold_{count}")

        # Train
        with open(f"{savedir}/Fold_{count}/train_list.txt", "w") as train_file:
            for tempIndex in trainIndex:
                subjectIndex = subjectList[tempIndex]
                for tempDir in dataDir:
                    if tempDir[:3] == subjectIndex and cross_subject_flag:  # 匹配id，选择数据
                        train_file.write(f"{dir}/{tempDir}\n")

                    if tempDir == subjectIndex and cross_subject_flag is False:
                        train_file.write(f"{dir}/{tempDir}\n")


        print(count, "Train  over")

        # Test
        with open(f"{savedir}/Fold_{count}/test_list.txt", "w") as test_file:
            for tempIndex in testIndex:
                subjectIndex = subjectList[tempIndex]
                for tempDir in dataDir:
                    if tempDir[:3] == subjectIndex and cross_subject_flag:  # 匹配id，选择数据
                        test_file.write(f"{dir}/{tempDir}\n")

                    if tempDir == subjectIndex and cross_subject_flag is False:
                        test_file.write(f"{dir}/{tempDir}\n")

        print(count, "Test  over")
        print(count, "Over")




def Matrix_computing(true_label, prediction_label):
    

    cm = confusion_matrix(true_label, prediction_label)
    
    # Macro-averaged metrics (average across all 4 classes)
    macro_recall = recall_score(true_label, prediction_label, average='macro')
    macro_precision = precision_score(true_label, prediction_label, average='macro')
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall + 1e-10)
    
    # Per-class recall (sensitivity)
    per_class_recall = recall_score(true_label, prediction_label, average=None)
    
    # Overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    report = classification_report(true_label, prediction_label, 
                                   target_names=["normal", "crackle", "wheeze", "both"])
    
    return cm, macro_recall, macro_precision, macro_f1, accuracy, report

    # confusion_matrix = metrics.confusion_matrix(true_label, prediction_label)

    # sen = (confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3]) / \
    #       (np.sum(confusion_matrix[1]) + np.sum(confusion_matrix[2]) + np.sum(confusion_matrix[3]))
    # spe = confusion_matrix[0][0] / np.sum(confusion_matrix[0])
    # ae = (sen + spe) / 2
    # hs = (2 * sen * spe) / (sen + spe)

    # report = classification_report(true_label, prediction_label, target_names=["normal", "crackle", "wheeze", "both"])

    # return confusion_matrix, sen, spe, ae, hs, report

def Matrix_computing_multilabel(true_labels, predictions):
    # true_labels: [N, 2] - [crackle, wheeze]
    # predictions: [N, 2] - [crackle, wheeze]
    
    # Convert back to 4-class for comparison
    def labels_to_class(labels):
        classes = []
        for label in labels:
            if label[0] == 0 and label[1] == 0:
                classes.append(0)  # Normal
            elif label[0] == 1 and label[1] == 0:
                classes.append(1)  # Crackle only
            elif label[0] == 0 and label[1] == 1:
                classes.append(2)  # Wheeze only
            else:  # label[0] == 1 and label[1] == 1
                classes.append(3)  # Both
        return classes
    
    true_classes = labels_to_class(true_labels)
    pred_classes = labels_to_class(predictions)
    
    # Standard metrics
    cm = confusion_matrix(true_classes, pred_classes)
    acc = accuracy_score(true_classes, pred_classes)
    
    # Per-label metrics (for crackle and wheeze independently)
    true_labels_np = true_labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    # Calculate sensitivity and specificity for each label
    sen_crackle = (true_labels_np[:, 0] * predictions_np[:, 0]).sum() / true_labels_np[:, 0].sum() if true_labels_np[:, 0].sum() > 0 else 0
    sen_wheeze = (true_labels_np[:, 1] * predictions_np[:, 1]).sum() / true_labels_np[:, 1].sum() if true_labels_np[:, 1].sum() > 0 else 0
    
    # Average sensitivity
    sen = (sen_crackle + sen_wheeze) / 2
    
    # Specificity
    spe_crackle = ((1 - true_labels_np[:, 0]) * (1 - predictions_np[:, 0])).sum() / (1 - true_labels_np[:, 0]).sum() if (1 - true_labels_np[:, 0]).sum() > 0 else 0
    spe_wheeze = ((1 - true_labels_np[:, 1]) * (1 - predictions_np[:, 1])).sum() / (1 - true_labels_np[:, 1]).sum() if (1 - true_labels_np[:, 1]).sum() > 0 else 0
    
    # Average specificity
    spe = (spe_crackle + spe_wheeze) / 2
    
    # Balanced accuracy
    ae = (sen + spe) / 2
    
    # Harmonic mean
    hs = (2 * sen * spe) / (sen + spe) if (sen + spe) > 0 else 0
    
    # Classification report (4-class view)
    report = classification_report(true_classes, pred_classes, 
                                   target_names=["normal", "crackle", "wheeze", "both"],
                                   zero_division=0)
    
    return cm, sen, spe, ae, hs, report



if __name__ == '__main__':
    CrossValidation(Config.preprocessed_dir_savesamples, Config.savedir_train_and_test, cross_subject_flag=True)
