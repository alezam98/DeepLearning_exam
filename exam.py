#!/usr/bin/env python
from UNet_model import generate_UNet_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os, glob, cv2
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval


TRAIN_BOOL = False
SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data(size=0.2, size_bool=True, print_bool=True):
    
    print('\nLoading data...')

    datasets = {}
    splits = ['train', 'test']

    for split in splits:
        
        filenames = np.load(f"data/filenames/{split}_filenames.npy")
        datasets[split] = {}
        datasets[split]['images'] = [] 
        datasets[split]['masks'] = []

        print(f"... for the {split} set:")
        for filename in tqdm(filenames, total=len(filenames)):
            image = Image.open(f'data/{split}_images/{filename}.jpg')
            np_image = np.array(image)
            datasets[split]['images'].append(np_image/255)
            
            image = Image.open(f'data/{split}_masks/{filename}_mask.gif')
            np_image = np.array(image)
            np_image = np.expand_dims(image, axis=2)
            datasets[split]['masks'].append(np_image)

        datasets[split]['images'] = np.array(datasets[split]['images'])
        datasets[split]['masks'] = np.array(datasets[split]['masks'])

    
    if size_bool:
        datasets['train']['images'], _, datasets['train']['masks'], _ = train_test_split(datasets['train']['images'], datasets['train']['masks'], test_size=1-size, shuffle=False)
        datasets['test']['images'], _, datasets['test']['masks'], _ = train_test_split(datasets['test']['images'], datasets['test']['masks'], test_size=1-size, shuffle=False)

    if print_bool:
        if size_bool: print(f'\nDatasets size (after resizing):')
        else: print(f'\nDatasets size:')
        print(f"train set: {datasets['train']['images'].shape[0]}")
        print(f"test set:  {datasets['test']['images'].shape[0]}")

        print(f'\nDatasets shape:')
        print(f"train set: {datasets['train']['images'].shape[1:]} (images) \t{datasets['train']['masks'].shape[1:]} (masks)")
        print(f"test set:  {datasets['test']['images'].shape[1:]} (images) \t{datasets['test']['masks'].shape[1:]} (masks)\n")

    return datasets['train']['images'], datasets['train']['masks'], datasets['test']['images'], datasets['test']['masks']



def plot_samples(train_images, train_masks, show_bool=True, save_bool=False):

    indexes = np.random.randint(low=0, high=len(train_images), size=9)

    fig = plt.figure(figsize=(12, 9))

    for i, index in enumerate(indexes):
        image = train_images[index]
        mask = train_masks[index]

        ax = plt.subplot(3, 3, i+1)
        ax.imshow(image)
        ax.imshow(mask, alpha=0.4, cmap='BuPu')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

    fig.tight_layout()

    if save_bool: plt.savefig('results/masked_samples.png', bbox_inches='tight', pad_inches=0.05)
    if show_bool: plt.show()



def print_best_parameters(best_params, save_bool=False):

    print("\nBest parameters")
    for key in best_params: print(f"{key}: {best_params[key]}")
    print()

    if save_bool:
        best_params_df = pd.DataFrame(best_params, index=[0])
        best_params_df.to_csv('results/parameters.csv', mode='w')
    



def plot_trials(trials, show_bool=True, save_bool=False):

    ys = [-t['result']['loss'] for t in trials.trials]
    best_index = np.argmax(np.array(ys))

    fig = plt.figure(figsize=(14, 5))
    
    ax = plt.subplot(1, 2, 1)
    xs = [t['tid'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]
    ax.set_xlim(xs[0]-1, xs[-1]+1)
    ax.scatter(xs, ys, s=20)
    ax.plot(xs[best_index], ys[best_index], 'ro', label='maximum accuracy')
    ax.set_xlabel('iteration')
    ax.set_ylabel('accuracy')
    ax.legend()

    ax = plt.subplot(1, 2, 2)
    xs = [t['misc']['vals']['learning_rate'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]
    ax.scatter(xs, ys, s=20)
    ax.plot(xs[best_index], ys[best_index], 'ro', label='best learning rate')
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('accuracy')
    ax.legend()

    fig.tight_layout()

    if save_bool: plt.savefig('results/trials.png', bbox_inches='tight', pad_inches=0.05)
    if show_bool: plt.show()



def get_IoU(images, masks, predictions):

    thresholds = np.linspace(0., 1., 101)
    IoU = []

    for threshold in thresholds:

        predictions_thresholded = predictions > threshold
        intersection = np.logical_and(masks, predictions_thresholded)
        union = np.logical_or(masks, predictions_thresholded)

        IoU.append( np.sum(intersection)/np.sum(union) )
    
    IoU = np.array(IoU)
    best_threshold, best_IoU = thresholds[np.argmax(IoU)], np.max(IoU)

    return thresholds, IoU, best_threshold, best_IoU



def plot_IoU(thresholds, IoU, best_threshold, best_IoU, show_bool=True, save_bool=False):

    fig = plt.figure(figsize=(7, 5))
    
    plt.plot(thresholds, IoU, linestyle='-')
    plt.plot(best_threshold, best_IoU, 'ro', label='best IoU')
    
    plt.plot([best_threshold, best_threshold], [0.5, best_IoU], 'k', linestyle='--')
    plt.text(best_threshold+0.02, 0.54, f'best IoU: {format(best_IoU, ".3f")}')
    plt.text(best_threshold+0.02, 0.5, f'best threshold: {format(best_threshold, ".2f")}')
    
    plt.ylabel('IoU')
    plt.xlabel('threshold')
    plt.legend()
    fig.tight_layout()
    
    if save_bool: plt.savefig('results/IoU_curve.png', bbox_inches='tight', pad_inches=0.05)
    if show_bool: plt.show()
    



def plot_metrics(history, metrics, show_bool=True, save_bool=False):
    
    fig = plt.figure(figsize=(14, 5))

    for i, metric in enumerate(metrics):      
        train_metrics = history[metric]
        val_metrics = history['val_'+metric]
        epochs = range(1, len(train_metrics) + 1)

        ax = plt.subplot(1, 2, i+1)
        
        ax.plot(epochs, train_metrics, linestyle="-", color="red", label="training")
        ax.plot(epochs, val_metrics, linestyle="-", color="blue", label="validation")
        
        if metric == 'loss': ax.set_ylabel("loss function")
        else: ax.set_ylabel("accuracy")
        ax.set_xlabel("epoch")
        ax.legend()

    fig.tight_layout()
    
    if save_bool: plt.savefig("results/loss_accuracy.png", bbox_inches='tight', pad_inches=0.05)
    if show_bool: plt.show()



def plot_predictions(test_images, test_masks, predictions, best_threshold, show_bool=True, save_bool=False):
    
    indexes = np.random.randint(low=0, high=len(test_images), size=3)
    pos = [[85, 85], [105, 90], [80, 90]]

    fig = plt.figure(figsize=(12, 8))

    for i, index in enumerate(indexes):
        image = test_images[index]
        mask = test_masks[index]
        prediction = (predictions[index] > best_threshold)
        good_pred = (prediction == mask).reshape(-1)
        good_pred = np.sum(good_pred)/len(good_pred)

        ax = plt.subplot(3, 3, 3*i+1)
        if i == 0: ax.set_title('Original masks')
        ax.imshow(image)
        ax.imshow(mask, alpha=0.4, cmap='BuPu')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ax = plt.subplot(3, 3, 3*i+2)
        if i == 0: ax.set_title('Predicted masks')
        ax.imshow(image)
        ax.imshow(prediction, alpha=0.4, cmap='BuGn')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        ax = plt.subplot(3, 3, 3*i+3)
        if i == 0: ax.set_title('Masks comparison')
        ax.imshow(mask, cmap='BuPu')
        ax.imshow(prediction, alpha=0.4, cmap='BuGn')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.text(pos[i][0], pos[i][1], f'{format(good_pred*100, ".3f")}%', fontsize=11, color='white')

        fig.tight_layout()

    if save_bool: plt.savefig(f'results/predictions.png', bbox_inches='tight', pad_inches=0.05)
    if show_bool: plt.show()



def get_agreement(test_masks, predictions, best_threshold, show_bool=True, save_bool=False):

    agreement = []

    for index, mask in enumerate(test_masks):
        
        prediction = (predictions[index] > best_threshold)
        good_pred = (prediction == mask).reshape(-1)
        good_pred = np.sum(good_pred)/len(good_pred)

        agreement.append(good_pred)

    return np.array(agreement)



def plot_agreement(test_masks, predictions, best_threshold, show_bool=True, save_bool=False):

    agreement = 100. * get_agreement(test_masks, predictions, best_threshold)
    mean_agreement = np.mean(agreement)
    percentages = np.linspace(np.min(agreement), 100., 81)

    fig = plt.figure(figsize=(7, 5))

    plt.hist(agreement, bins=percentages, density=True)
    plt.plot([mean_agreement, mean_agreement], [0, 3.5], linestyle='--', color='black')
    plt.scatter([mean_agreement], [3.5], color='black')

    plt.ylabel('normalized distribution')
    plt.xlabel('agreement percentage')

    plt.text(mean_agreement-1., 3.4, f'mean agreement:')
    plt.text(mean_agreement-1., 3.25, f'{format(mean_agreement, ".3f")}%')

    if save_bool: plt.savefig(f'results/agreement.png', bbox_inches='tight', pad_inches=0.05)
    if show_bool: plt.show()







def main():
    
    # LOADING DATA
    train_images, train_masks, test_images, test_masks = load_data()
    plot_samples(train_images, train_masks, save_bool=True)


    # TRAINING
    if TRAIN_BOOL:

        # hyperparameter search
        def objective(parameters):
            model = generate_UNet_model(parameters)
            model.fit(
                train_images, train_masks,
                validation_split = 0.3,
                batch_size = 16,
                epochs = 30,
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
            )

            test_loss, test_acc = model.evaluate(test_images, test_masks)
            return {'loss': -test_acc, 'status': STATUS_OK, 'eval_time': time.time()}
        
        max_evals = 50
        search_space = {
            "learning_rate": hp.loguniform("learning_rate", -10, -3)
        }
        trials = Trials()
        algorithm = tpe.suggest
        best = fmin(objective, space=search_space, algo=algorithm, max_evals=max_evals, trials=trials)

        # best parameters
        best_params = space_eval(search_space, best)
        print_best_parameters(best_params, save_bool=True)
        plot_trials(trials, save_bool=True)


        # training best model
        model = generate_UNet_model(best_params)
        history = model.fit(
            train_images, train_masks,
            validation_split = 0.3,
            batch_size = 16,
            epochs = 50,
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
        )

        history_df = pd.DataFrame(history.history)
        history_df.to_csv('results/history.csv', mode='w')
    
        model.save('./model')

    else:

        history = dict(pd.read_csv('results/history.csv'))
        model = tf.keras.models.load_model('./model')

    
    # MODEL EVALUATION
    print('\nEvaluating model...')

    loss, accuracy = model.evaluate(test_images, test_masks)
    print('\nModel metrics evaluated on the test set')
    print(f'test loss:     {format(loss, ".3f")}')
    print(f'test accuracy: {format(accuracy, ".3f")}\n')
    with open('results/test_metrics.dat', 'w') as fp:
        fp.write(f'{loss}\n')
        fp.write(f'{accuracy}\n')

    predictions = model.predict(test_images)
    thresholds, IoU, best_threshold, best_IoU = get_IoU(test_images, test_masks, predictions)

    plot_agreement(test_masks, predictions, best_threshold, save_bool=True)
    plot_IoU(thresholds, IoU, best_threshold, best_IoU, save_bool=True)
    if TRAIN_BOOL: plot_metrics(history.history, ['loss', 'accuracy'], save_bool=True)
    else: plot_metrics(history, ['loss', 'accuracy'], save_bool=True)
    plot_predictions(test_images, test_masks, predictions, best_threshold, save_bool=True)



if __name__ == "__main__":
    main()
