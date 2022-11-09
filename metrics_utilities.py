# Collection of metrics utilities
#   cm_cr - 
#   plot_conf_mat_w_and_wo_norm -

# Import dependencies
import pandas as pd
import numpy as np
# Matplotlib for visualization
from matplotlib import pyplot as plt
# Seaborn for easier visualization
import seaborn as sns
sns.set_style('darkgrid')
# To display dataframes side by side
from IPython.display import display_html 
# Metrics
from sklearn.metrics import confusion_matrix, classification_report


# Function to display confusion matrix dataframes and classification report
def cm_cr(model_name, y_test, y_pred, target_names, cr=True):
    """ Disolay confusion matrix dataframes with and without normalization
        and print classification report if cr = True

    Args:
        model_name: name of the model
        y_test: test target variable
        y_pred: prdiction
        target_names: list of target class names
        cr: print classification report if True - default

    Returns:
        Display confusion matrix dataframes side by side
        and classification report if selected (default)
    """
    
    # Print header
    print('\t\t\t', model_name)
    print('\t\t\t', '='*len(model_name))
    
    # Create dataframe for confusion matrix for y_test and y_pred
    cm = confusion_matrix(y_test, y_pred)
    conf_df = pd.DataFrame(cm, columns=target_names, index=target_names)
    conf_df.index.name = 'True Labels'
    conf_df = conf_df.rename_axis('Predicted Labels', axis='columns')
    
    # Dataframe for normalizwzed confusion matrix
    cm = np.around(cm / cm.sum(axis=1)[:, np.newaxis], 2)
    conf_dfn = pd.DataFrame(cm, columns=target_names, index=target_names)
    conf_dfn.index.name = 'True Labels'
    conf_dfn = conf_dfn.rename_axis('Predicted Labels', axis='columns')
  
    # Display dataframes side by side
    conf_df_styler = conf_df.style.set_table_attributes("style='display:inline'").set_caption('Confusion Matrix')
    conf_dfn_styler = conf_dfn.style.set_table_attributes("style='display:inline'").set_caption('Normalized Confusion Matrix').format(precision=2)
    
    space = "\xa0" * 15
    display_html(conf_df_styler._repr_html_() + space + conf_dfn_styler._repr_html_(), raw=True)
    
    if cr:
        # Display classification report
        print()
        print(classification_report(y_test, y_pred, target_names=target_names))
    print()
    
    
# ==============

# Function to plot 
def plot_conf_mat_w_and_wo_norm(model_name, y_test, y_pred, target_names, color):
    # Plot confusion matrix heatmaps

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.suptitle(model_name, fontsize=14)
    f.subplots_adjust(top=0.85, wspace=0.3)

    # confusion matrix without normalization
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat,
                annot=True,
                annot_kws=dict(fontsize=14),
                fmt='d',
                cbar=True,
                square=True,
                cmap=color,
                linecolor='red',
                linewidth=0.01,
                ax=ax1)

    ax1.set_xticklabels(labels=target_names)
    ax1.set_yticklabels(labels=target_names, va='center')
    ax1.set_title('Confusion Matrix w/o Normalization')
    ax1.set_xlabel('Predicted Label', size=12)
    ax1.set_ylabel('True Label', size=12)

    # normalized confusion matrix
    matn = mat / mat.sum(axis=1)[:, np.newaxis]
    sns.heatmap(matn,
                annot=True,
                annot_kws=dict(fontsize=14),
                fmt='.2f',
                cbar=True,
                square=True,
                cmap=color,
                linecolor='red',
                linewidth=0.01,
                vmin = 0,
                vmax = 1,
                ax=ax2)

    ax2.set_xticklabels(labels=target_names)
    ax2.set_yticklabels(labels=target_names, va='center')
    ax2.set_title('Normalized Confusion Matrix')
    ax2.set_xlabel('Predicted Label', size=12)
    ax2.set_ylabel('True Label', size=12)

    plt.show()