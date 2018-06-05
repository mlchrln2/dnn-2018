from __future__ import print_function
import keras
keras.__version__
import os.path

j=1
while os.path.exists('../figures/analysis_network'+str(j)+'.png'):
    j+=1
network_path = "../networks/network"+str(j)+'/'

keras_model = keras.models.load_model(network_path+"nt3_network"+str(j)+".h5")
keras_model.summary()

import deeplift
from deeplift.blobs import NonlinearMxtsMode
from deeplift.conversion import keras_conversion as kc

#Three different models, one each for RevealCancel, Gradient and GuidedBackprop
revealcancel_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)
grad_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.Gradient)
guided_backprop_model = kc.convert_sequential_model(model=keras_model, nonlinear_mxts_mode=NonlinearMxtsMode.GuidedBackprop)

### load data
import pandas as pd
import numpy as np
from keras.utils import np_utils
df_test = (pd.read_csv('../data-05-31-2018/formatted_full_data.csv',header=None).values).astype('float32')
df_y_test = df_test[:,0].astype('int')
seqlen = df_test.shape[1]
Y_test = np_utils.to_categorical(df_y_test,2)
X_test = df_test[:, 1:seqlen].astype(np.float32)

### import deeplift and compile functions
from deeplift.util import compile_func
import numpy as np
from keras import backend as K

deeplift_model = revealcancel_model
deeplift_prediction_func = compile_func([deeplift_model.get_layers()[0].get_activation_vars()],
                                       deeplift_model.get_layers()[-1].get_activation_vars())
original_model_predictions = keras_model.predict(X_test, batch_size=200)
converted_model_predictions = deeplift.util.run_function_in_batches(
                                input_data_list=[X_test],
                                func=deeplift_prediction_func,
                                batch_size=200,
                                progress_update=None)
print("difference in predictions:",np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)))
assert np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)) < 10**-5
predictions = converted_model_predictions


### specify layers which we want to backprop with different methods
### find_scores_layer_idx=0 means scores for input layer,
### target_layer_idx=-2 for nonlinear softmax and sigmoid outputs
from keras import backend as K
import deeplift
from deeplift.util import get_integrated_gradients_function

revealcancel_func = revealcancel_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-2)
grad_times_inp_func = grad_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-2)
guided_backprop_times_inp_func = guided_backprop_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-2)

gradient_func = grad_model.get_target_multipliers_func(find_scores_layer_idx=0, target_layer_idx=-2)
guided_backprop_func = guided_backprop_model.get_target_multipliers_func(find_scores_layer_idx=0, target_layer_idx=-2)

#pure-gradients or pure-guidedbackprop perform rather poorly because they produce scores on pixels that are 0 (which are
#the backround in MNIST). But we can give them a slight advantage by masking out positions that
#are zero. Also, the method of simonyan et al uses the magnitude of the gradient.
simonyan_func_masked = lambda input_data_list, **kwargs: ((input_data_list[0]>0.0)*
                        np.abs(np.array(gradient_func(input_data_list=input_data_list,**kwargs))))
guided_backprop_func_masked = lambda input_data_list, **kwargs: ((input_data_list[0]>0.0)*
                               guided_backprop_func(input_data_list=input_data_list, **kwargs))

#prepare the integrated gradients scoring function
#heads-up: these take 5x and 10x longer to compute respectively!
integrated_grads_5 = get_integrated_gradients_function(gradient_func, 5)
integrated_grads_10 = get_integrated_gradients_function(gradient_func, 10)


#input_test = (pd.read_csv('/Users/Zireael/Desktop/Maslov/argonne/feature_importance/mean_normal_baseline.csv',header=None).values).astype('float32')
input_test = np.mean(X_test[df_y_test == 0], axis=0)
#X_ref = input_test[:, 1:seqlen].astype(np.float32)
input_test.ravel()


### compute scores for 0(normal) and 1(tumor)
from collections import OrderedDict
method_to_task_to_scores = OrderedDict()
print("HEADS UP! integrated_grads_5 and integrated_grads_10 take 5x and 10x longer to run respectively")
print("Consider leaving them out to get faster results")
for method_name, score_func in [
                               ('revealcancel', revealcancel_func),
                               ('guided_backprop_masked', guided_backprop_func_masked),
                               ('guided_backprop_times_inp', guided_backprop_times_inp_func),
                               ('simonyan_masked', simonyan_func_masked),
                               ('grad_times_inp', grad_times_inp_func),
                               ('integrated_grads_5', integrated_grads_5),
                               ('integrated_grads_10', integrated_grads_10)
]:
    print("Computing scores for:",method_name)
    method_to_task_to_scores[method_name] = {}
    for task_idx in range(2):
        print("\tComputing scores for task: "+str(task_idx))
        scores = np.array(score_func(
                    task_idx=task_idx,
                    input_data_list=[X_test],
                    #input_references_list=[np.zeros_like(X_test)],
                    input_references_list=input_test,
                    batch_size=1000,
                    progress_update=None))
        method_to_task_to_scores[method_name][task_idx] = scores


### save scores
index = ['Row'+str(i) for i in range(1, len(scores)+1)]
df = pd.DataFrame(scores, index=index)
#scores.to_csv("scores.csv", sep='\t')


index = ['Row'+str(i) for i in range(1, len(method_to_task_to_scores['revealcancel'][0])+1)]
df = pd.DataFrame(method_to_task_to_scores['revealcancel'][0], index=index)
df.to_csv(network_path+"revealcancel_0_scores_mean.csv", sep='\t')

index = ['Row'+str(i) for i in range(1, len(method_to_task_to_scores['revealcancel'][1])+1)]
df = pd.DataFrame(method_to_task_to_scores['revealcancel'][1], index=index)
df.to_csv(network_path+"revealcancel_1_scores_mean.csv", sep='\t')
