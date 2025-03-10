import matplotlib.pyplot as plt
import numpy as np

def final_rrs_histogram(final_rrs):
    #plt.figure(figsize=(3, 2))
    plt.hist(final_rrs, bins=np.arange(0, 1.1, 0.05), color=[0.12710496, 0.64018454, 0.60749712, 1.])
    plt.xlabel('intervention final layer reciprocal rank', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    #plt.title('Intervention Final Reciprocal Ranks')
    #plt.show()
    
def plot_intervention(wrapper, mlps, word_answer, text, mlp_O, layer_start,
    layer_end=None,
):
    if layer_end is None:
        layer_end = wrapper.model.config.n_layer
    
    tokens_answer = wrapper.tokenizer.encode(f"{word_answer}")
    if len(tokens_answer) > 1:
        print(f'"{word_answer}"', tokens_answer, 
            [wrapper.tokenizer.decode([t]) for t in tokens_answer]
        )

    wrapper.reset_mlps(mlps)
    ids, toks = wrapper.tokenize_ids_str(text)
    cntrl_logits = wrapper.get_layers(ids)
    cntrl_rrs = wrapper.rr_per_layer(cntrl_logits, word_answer)

    wrapper.o_intervention(mlp_O.unsqueeze(0), layer_start=layer_start, layer_end=layer_end)

    interv_logits = wrapper.get_layers(ids)
    interv_rrs = wrapper.rr_per_layer(interv_logits, word_answer)

    plot_rrs_before_and_after_intervention(wrapper, cntrl_rrs, interv_rrs, title=text, 
        label=word_answer, layer_start=layer_start, layer_end=layer_end)

def plot_accuracy(correct_cntrl_accuracy, correct_interv_accuracy):
    """
    Parameters:
        correct_cntrl_accuracy: float, accuracy scores for control condition
        correct_interv_accuracy: float, accuracy scores for intervention condition
    """
    bars = plt.bar(['Control', 'Intervention'], [correct_cntrl_accuracy, correct_interv_accuracy])
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy', fontsize=10)
    
    # Add text annotions above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    #plt.show()

def plot_intervention_mean(wrapper, mlps, word_answers, word_args, mlp_O, 
    layer_start, layer_end=None,
):
    """
    Plot mean reciprocal ranks 1) argument and answer, 2) answer before and after intervention
    Plot histogram of the final reciprocal rank under intervention
    Plot mean accuracy of the intervention

    Parameters:
    word_answers: list of answers 
    word_args: list of arguments 
    mlp_O: mlp output vector to use for intervention 
    layer_start: layer to start intervening on 
    layer_end: layer to end intervening before (default is n_layer),  not including the layer_end 
    """
    if layer_end is None:
        layer_end = wrapper.model.config.n_layer

    interv_rrs = np.empty((len(word_answers), wrapper.model.config.n_layer+1))
    interv_rrs_arg = np.empty((len(word_answers), wrapper.model.config.n_layer+1))
    cntrl_rrs = np.empty((len(word_answers), wrapper.model.config.n_layer+1))
    cntrl_pred_idxs = np.empty((len(word_answers)), dtype=int)
    interv_pred_idxs = np.empty((len(word_answers)), dtype=int)
    answer_ids = []

    for i, (word_arg, word_answer) in enumerate(zip(word_args, word_answers)):
        text = f"returning ensus machine{word_arg} returning ensus machine{word_arg} returning ensus machine"
        tokens_answer = wrapper.tokenizer.encode(f"{word_answer}")
        answer_ids.append(tokens_answer[0])
        if len(tokens_answer) > 1:
            print(f'"{word_answer}"', tokens_answer, [wrapper.tokenizer.decode([t]) for t in tokens_answer])

        wrapper.reset_mlps(mlps)
        ids, toks = wrapper.tokenize_ids_str(text)
        cntrl_logit = wrapper.get_layers(ids)
        cntrl_rr = wrapper.rr_per_layer(cntrl_logit, word_answer)

        wrapper.o_intervention(mlp_O.unsqueeze(0), layer_start=layer_start, layer_end=layer_end)
        interv_logit = wrapper.get_layers(ids)

        interv_rr = wrapper.rr_per_layer(interv_logit, word_answer)
        interv_rr_arg = wrapper.rr_per_layer(interv_logit, word_arg)

        # reciprocal ranks  
        interv_rrs[i] = interv_rr
        cntrl_rrs[i] = cntrl_rr
        interv_rrs_arg[i] = interv_rr_arg

        # prediction 
        cntrl_pred_idx = cntrl_logit[-1].argmax(dim=-1)
        interv_pred_idx = interv_logit[-1].argmax(dim=-1)
        cntrl_pred_idxs[i] = cntrl_pred_idx
        interv_pred_idxs[i] = interv_pred_idx

    # calculate the mean of the rrs
    cntrl_rrs_mean = np.mean(cntrl_rrs, axis=0)
    interv_rrs_mean = np.mean(interv_rrs, axis=0)
    interv_rrs_arg_mean = np.mean(interv_rrs_arg, axis=0)
    final_rrs = interv_rrs[:, -1]
    final_rrs_below = [word_args[i] for i in range(len(final_rrs)) if final_rrs[i] < 0.95]
    final_rrs_above = [word_args[i] for i in range(len(final_rrs)) if final_rrs[i] >= 0.95]
    print(f"final rrs below 0.95: {final_rrs_below}")
    print(f"final rrs above 0.95: {final_rrs_above}")

    # accuracy 
    #answer_ids = tokenizer.encode(word_answers)
    correct_cntrl = [i for i in range(len(cntrl_pred_idxs)) if cntrl_pred_idxs[i] == answer_ids[i]]
    correct_interv = [i for i in range(len(interv_pred_idxs)) if interv_pred_idxs[i] == answer_ids[i]]
    correct_cntrl_accuracy = len(correct_cntrl) / len(answer_ids)
    correct_interv_accuracy = len(correct_interv) / len(answer_ids)
    wrong_interv_words = [word_args[i] for i in range(len(interv_pred_idxs)) 
        if interv_pred_idxs[i] != answer_ids[i]]
    correct_interv_words = [word_args[i] for i in correct_interv]
    print(f"wrong interv words: {wrong_interv_words}")
    print(f"correct interv words: {correct_interv_words}")

    # plot 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 4))
    fig.suptitle(f"Intervention", fontsize=12)

    plt.sca(ax1)
    plot_rrs_before_and_after_intervention(wrapper, cntrl_rrs_mean, interv_rrs_mean, 
        layer_start=layer_start, layer_end=layer_end, mean=True,
    )
    plt.sca(ax2)
    plot_rrs_aug_answer(wrapper, aug_rrs=interv_rrs_arg_mean, answer_rrs=interv_rrs_mean,
        layer_start=layer_start, layer_end=layer_end, mean=True,
    )
    plt.sca(ax3) 
    final_rrs_histogram(final_rrs)
    plt.sca(ax4)
    plot_accuracy(correct_cntrl_accuracy, correct_interv_accuracy)
    
    plt.tight_layout()

def plot_rrs_aug_answer(wrapper, aug_rrs, answer_rrs, mean=False,
    layer_start=None, layer_end=None,
):
    if layer_end is None:
        layer_end = wrapper.model.config.n_layer
        
    colors = [
        [0.12710496, 0.44018454, 0.70749712, 1.        ], #Arg
        [0.12710496, 0.64018454, 0.60749712, 1.        ], #Answer
    ]
    #plt.title(title, fontsize=18)
    plt.xticks(np.arange(wrapper.model.config.n_layer), 
        np.arange(wrapper.model.config.n_layer), rotation=90)
    plt.plot(aug_rrs[1:], label='augument', marker='o', color=colors[0]) #Why [1:]? rrs[0] is the embedding table step
    plt.plot(answer_rrs[1:], label='answer', marker='o', color=colors[1]) #Why [1:]? rrs[0] is the embedding table step
    if layer_start is not None:
        plt.axvspan(layer_start, layer_end-1, facecolor='.92')
    plt.legend(fontsize=8)
    plt.xlabel("Layer", fontsize=10)
    plt.ylabel("Mean Reciprocal Rank" if mean else "Reciprocal Rank", fontsize=10)
    plt.yticks([0.,.2,.4,.6,.8,1.], ['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.tight_layout()
    
def plot_rrs_before_and_after_intervention(wrapper, before_rrs, after_rrs, 
    title='', label='', layer_start=None, layer_end=None, mean=False,
):
    if layer_end is None:
        layer_end = wrapper.model.config.n_layer
    
    colors = [
        [0, 0, 0, 1], # zero shot
        [0.12710496, 0.64018454, 0.60749712, 1.        ], # + mlp_O
    ]
    plt.title(title, fontsize=10)
    plt.xticks(np.arange(wrapper.model.config.n_layer), 
        np.arange(wrapper.model.config.n_layer), rotation=90)
    plt.plot(before_rrs[1:], label=f'{label}: zero shot', marker='o', color=colors[0]) #Why [1:]? rrs[0] is the embedding table step
    plt.plot(after_rrs[1:], label=f'{label}:'+' + mlp_O', marker='o', color=colors[1]) #Why [1:]? rrs[0] is the embedding table step

    if layer_start is not None:
        plt.axvspan(layer_start, layer_end-1, facecolor='.92')
    plt.legend(fontsize=8)
    plt.xlabel("Layer", fontsize=10)
    plt.ylabel("Mean Reciprocal Rank" if mean else "Reciprocal Rank", fontsize=10)
    plt.ylim(0, 1.1)
    plt.yticks([0.,.2,.4,.6,.8,1.], ['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.tight_layout()