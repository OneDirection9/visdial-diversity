import os
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import nltk
from nltk.util import ngrams

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import DataLoader
import math
from six.moves import range
import matplotlib.pyplot as plt
from eval_utils.rank_questioner import rankQBot
from collections import Counter
from scipy.stats import entropy

def dialogDump(params,
               dataset,
               split,
               aBot,
               qBot=None,
               beamSize=1,
               saveFolder="dialog_output"):
    '''
        Generates dialog and saves it to a json for later visualization.
        If only A-Bot is given, A-Bot answers are generated given GT image,
        caption and questions. If both agents are given, dialog is generated
        by both agents conversing (A-Bot is shown the GT image and both
        agents have access to a caption generated by a pre-trained captioning
        model).

        Arguments:
            params  : Parameter dict for all options
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'
            aBot    : A-Bot
            qBot    : Q-Bot (Optional)

            beamSize : Beam search width for generating utterrances
            saveFolder : Folder path for saving dialog related files
    '''
    text, dialog_metrics = run_dialog(params,
               dataset,
               split,
               aBot,
               qBot,
               beamSize=beamSize)

    dist_1 = dialog_metrics["dist_1"]
    dist_2 = dialog_metrics["dist_2"]
    dist_1_CI = dialog_metrics["dist_1_CI"]
    dist_2_CI = dialog_metrics["dist_2_CI"]

    average_precision_CI = dialog_metrics["average_precision_CI"]
    ent_1_CI = dialog_metrics["ent_1_CI"]
    ent_2_CI = dialog_metrics["ent_2_CI"]
    unique_questions_CI = dialog_metrics["unique_questions_CI"]
    mutual_overlap_CI = dialog_metrics["mutual_overlap_CI"]

    unique_questions = dialog_metrics["tot_unique_questions"]
    tot_examples = dialog_metrics["tot_examples"]
    mean_unique_questions = dialog_metrics["mean_unique_questions"]
    std_unique_questions = dialog_metrics["std_unique_questions"]

    similarity_scores_mean = dialog_metrics["similarity_scores_mean"]
    norm_difference_scores_mean = dialog_metrics["norm_difference_scores_mean"]
    norm_scores_mean = dialog_metrics["norm_scores_mean"]
    huber_scores_mean = dialog_metrics["huber_scores_mean"]

    average_precision = dialog_metrics["average_precision"]
    per_round_precision = dialog_metrics["per_round_precision"]

    bleu_metric = dialog_metrics["mutual_overlap_score"]
    novel_questions = dialog_metrics["tot_novel_questions"]
    avg_novel_questions = dialog_metrics["avg_novel_questions"]
    tot_questions = dialog_metrics["tot_questions"]

    nll = dialog_metrics['NLL']

    ent_1 = dialog_metrics["ent_1"]
    ent_2 = dialog_metrics["ent_2"]

    savePathJson = os.path.join(saveFolder,"results.json")
    saveMetricsFile = os.path.join(saveFolder,"metrics.txt")
    saveLatexFile = os.path.join(saveFolder,"latex.txt")

    with open(saveMetricsFile, "w") as fp:
        fp.write("Metrics: \n")

    with open(saveMetricsFile, "w") as fp:

        print("Writing dialog metrics data to file: {}".format(saveMetricsFile))

        fp.write("tot unique questions: %d"%unique_questions + "\n")
        fp.write("tot examples: %d"%tot_examples + "\n")
        fp.write("avg unique questions per example: %f"%mean_unique_questions + "\n")
        fp.write("std unique questions per example: %f"%std_unique_questions + "\n")

        fp.write("Mutual Overlap: %f"%bleu_metric + "\n")
        fp.write("Ent-1: %f"%ent_1 + "\n")
        fp.write("Ent-2: %f"%ent_2 + "\n")
        fp.write("Dist-1: %f"%dist_1 + "\n")
        fp.write("Dist-2: %f"%dist_2 + "\n")

        fp.write("novel questions: %d" % novel_questions + "\n")
        fp.write("avg novel questions: %f" % avg_novel_questions + "\n")
        fp.write("tot_questions: %d" % tot_questions + "\n")
        fp.write("average precision for questions: %f"%average_precision + "\n")
        fp.write("nll of GT questions: %f"%nll + "\n")

        fp.write("Mutual Overlap CI: %f"% mutual_overlap_CI  +"\n")
        fp.write("Average Precision CI: %f"% average_precision_CI + "\n")
        fp.write("Unique Question CI: %f"% unique_questions_CI + "\n")
        fp.write("Ent-1-CI: %f"% ent_1_CI + "\n")
        fp.write("Ent-2-CI: %f"% ent_2_CI + "\n")
        fp.write("Dist-1-CI: %f"% dist_1_CI + "\n")
        fp.write("Dist-2-CI: %f"% dist_2_CI + "\n")


        fp.write("cos similarity between consecutive rounds \n")
        fp.write(",".join(map(str,similarity_scores_mean)) + "\n")

        fp.write("difference of norms between consecutive rounds \n")
        fp.write(",".join(map(str,norm_difference_scores_mean)) + "\n")

        fp.write("mean norm at each round \n")
        fp.write(",".join(map(str,norm_scores_mean)) + "\n")

        fp.write("huber loss between consecutive rounds \n")
        fp.write(",".join(map(str,huber_scores_mean)) + "\n")

        fp.write("round to round precision for questions \n")
        fp.write(",".join(map(str,per_round_precision)) + "\n")

    with open(savePathJson, "w") as fp:
        print("Writing dialog text data to file: {}".format(savePathJson))
        json.dump(text, fp)

    # with open(saveMetricsJson, "w") as fp:
    #     print("Writing dialog metrics to file: {}".format(saveMetricsJson))
    #     json.dump(dialog_metrics, fp)

    # write latex string
    latex_code =  " $ " + str(round(novel_questions,2)) + " $ " + " & " + " $ " + str(round(mean_unique_questions,2))  + " $ " + " \pm " + str(round(unique_questions_CI,2)) \
                 + " $ \pm $ " + " $ " + str(round(bleu_metric,2)) + " $ " + " $ \pm $ " +  " $ " + str(round(mutual_overlap_CI,2)) + " $ " + " & " +  " $ " +str(round(ent_1,2)) \
                 + " $ " + " $\pm$ " + " $ " + str(round(ent_1_CI,2)) + " $ " + " & " + " $ " + str(round(ent_2,2)) \
                 + " $ " + " $\pm$ " +  " $ " + str(round(ent_2_CI,2)) +  " $ " +\
                  "& $" + str(round(dist_1,2)) + " $ &" + "$ \pm $" + "& $" + str(round(dist_1_CI,2)) + " $ " + \
                  "& $" + str(round(dist_2, 2)) + " $ &" + "$ \pm $" + "& $" + str(round(dist_2_CI, 2)) + " $ " + \
    " && " + " $ " + str(round(nll,2)) + " $ " + " & " + " $ " + str(round(average_precision,2)) \
                 + " $ " + " $ \pm$ " + " $ " +  str(round(average_precision_CI,2)) + " $ "

    with open(saveLatexFile, "w") as fp:
        print("Writing latex code to file: {}".format(saveLatexFile))
        fp.write(latex_code)

    print("Done!")

    fig = plt.figure()
    plt.plot(similarity_scores_mean, label='Cos')
    plt.plot(norm_difference_scores_mean, label='Norm Penalty')
    plt.plot(huber_scores_mean, label='Huber')
    plt.title('Similarity of consecutive embeddings')
    plt.ylabel('Similarity')
    plt.xlabel("Round")
    plt.legend()
    fig.savefig(os.path.join(saveFolder,'Similarity_Metrics_Plot.png'))

    fig = plt.figure()
    plt.plot(norm_scores_mean)
    plt.title('Norm vs Round')
    plt.ylabel('Norm')
    plt.xlabel("Round")
    plt.legend()
    fig.savefig(os.path.join(saveFolder,'norms.png'))

def run_dialog(params,
               dataset,
               split,
               aBot,
               qBot=None,
               beamSize=1):

    assert aBot is not None or (qBot is not None and aBot is not None),\
                            "Must provide either an A-Bot alone or both \
                            Q-Bot and A-Bot when generating dialog"
    # rankMetrics, _ = rankQBot(qBot, dataset, 'val')
    rankMetrics = None

    old_split = dataset.split
    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    train_questions = set()

    dataset.split = 'train'
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    ind2word = dataset.ind2word
    to_str_gt = lambda w: str(" ".join([ind2word[x] for x in filter(lambda x:\
                    x>0,w.data.cpu().numpy())])) #.encode('utf-8','ignore')
    to_str_pred = lambda w, l: str(" ".join([ind2word[x] for x in list( filter(
        lambda x:x>0,w.data.cpu().numpy()))][:l.data.cpu()[0]])) #.encode('utf-8','ignore')

    for idx, batch in enumerate(dataloader):
        # append all questions in train in a set to calculate downstream metrics
        gtQuestions = Variable(batch['ques'], requires_grad=False)
        gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
        if gtQuesLens.shape[0] < batchSize:
            break

        # iterate through the batch and add to dictionary
        for j in range(batchSize):
            for rnd in range(numRounds):
                question_str = to_str_pred(gtQuestions[j,rnd,:], gtQuesLens[j,rnd])
                train_questions.add(question_str[8:])

    print("train questions len:", len(train_questions))

    dataset.split = split

    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    text = {'data': []}
    if '%s_img_fnames' % split not in dataset.data.keys():
        print("[Error] Need coco directory and info as input " \
               "to -cocoDir and -cocoInfo arguments for locating "\
               "coco image files.")
        print("Exiting dialogDump without saving files.")
        return None

    getImgFileName = lambda x: dataset.data['%s_img_fnames' % split][x]
    getImgId = lambda x: int(getImgFileName(x)[:-4][-12:])

    similarity_scores_mean = Variable(torch.zeros(numRounds))
    norm_difference_scores_mean = Variable(torch.zeros(numRounds))
    norm_scores_mean = Variable(torch.zeros(numRounds))
    huber_scores_mean = Variable(torch.zeros(numRounds))

    if params["useGPU"]:

        similarity_scores_mean = similarity_scores_mean.cuda()
        norm_difference_scores_mean = norm_difference_scores_mean.cuda()
        norm_scores_mean = norm_scores_mean.cuda()
        huber_scores_mean = huber_scores_mean.cuda()

    tot_idx = 0
    output_dialog = True
    tot_examples = 0
    unique_questions = 0
    unique_questions_list = []
    mutual_overlap_list = []
    ent_1_list = []
    ent_2_list = []
    dist_1_list = []
    dist_2_list = []
    avg_precision_list = []

    bleu_metric = 0
    novel_questions = 0
    oscillating_questions_cnt = 0
    per_round_bleu = np.zeros(numRounds)
    ent_1 = 0
    ent_2 = 0

    for idx, batch in enumerate(dataloader):
        print("current batch:",idx)
        if idx > 3:
            output_dialog = False
        tot_idx = tot_idx + 1
        imgIds = [getImgId(x) for x in batch['index']]
        dialog = [{'dialog': [], 'image_id': imgId} for imgId in imgIds]

        if dataset.useGPU:
            batch = {key: v.cuda() if hasattr(v, 'cuda')\
                else v for key, v in batch.items()}

        image = Variable(batch['img_feat'], volatile=True)
        caption = Variable(batch['cap'], volatile=True)
        # ignoring the last batch
        if caption.size()[0] < batchSize:
            break
        captionLens = Variable(batch['cap_len'], volatile=True)
        if qBot is None:  # A-Bot alone needs ground truth dialog
            gtQuestions = Variable(batch['ques'], volatile=True)
            gtQuesLens = Variable(batch['ques_len'], volatile=True)
            gtAnswers = Variable(batch['ans'], volatile=True)
            gtAnsLens = Variable(batch['ans_len'], volatile=True)

        if aBot:
            aBot.eval(), aBot.reset()
            aBot.observe(
                -1, image=image, caption=caption, captionLens=captionLens)
        if qBot:
            qBot.eval(), qBot.reset()
            qBot.observe(-1, caption=caption, captionLens=captionLens)
        questions = []

        for j in range(batchSize):
            caption_str = to_str_gt(caption[j])[8:-6]
            dialog[j]['caption'] = caption_str
        past_dialog_hidden = None
        cur_dialog_hidden = None
        question_str_list = [[] for _ in range(batchSize)]
        gt_questions_str = [[] for _ in range(batchSize)]

        gtQuestions = Variable(batch['ques'], volatile=True)
        gtQuesLens = Variable(batch['ques_len'], volatile=True)
        gtAnswers = Variable(batch['ans'], volatile=True)
        gtAnsLens = Variable(batch['ans_len'], volatile=True)

        for round in range(numRounds):

            if aBot is not None and qBot is None:
                aBot.observe(
                    round,
                    ques=gtQuestions[:, round],
                    quesLens=gtQuesLens[:, round])
                aBot.observe(
                    round,
                    ans=gtAnswers[:, round],
                    ansLens=gtAnsLens[:, round])
                _ = aBot.forward()
                answers, ansLens = aBot.forwardDecode(
                    inference='greedy', beamSize=beamSize)

            elif aBot is not None and qBot is not None:
                questions, quesLens = qBot.forwardDecode(
                    beamSize=beamSize, inference='greedy')
                qBot.observe(round, ques=questions, quesLens=quesLens)
                aBot.observe(round, ques=questions, quesLens=quesLens)
                answers, ansLens = aBot.forwardDecode(
                    beamSize=beamSize, inference='greedy')
                aBot.observe(round, ans=answers, ansLens=ansLens)
                qBot.observe(round, ans=answers, ansLens=ansLens)
                qBot.encoder()

            cur_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]
            if round == 0:
                past_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            similarity_scores = cos(cur_dialog_hidden, past_dialog_hidden)
            norm_difference_scores = torch.abs(torch.norm(cur_dialog_hidden, p=2, dim=1) - \
                          torch.norm(past_dialog_hidden,p=2,dim=1))
            # calculate norm
            norm_scores = torch.norm(cur_dialog_hidden, p=2, dim=1)
            # calculate Huber Loss/ Difference at consecutive rounds with Huber Threshold = 0.1
            threshold = 0.1
            norm_differences = torch.abs(cur_dialog_hidden - past_dialog_hidden)
            l2_mask = norm_differences <= threshold
            norm_differences_new = 0.5 * norm_differences * norm_differences * (l2_mask == 1).float()
            l1_mask = norm_differences > threshold
            norm_differences_new = norm_differences_new + (((l1_mask == 1).float()) * (threshold *
                                                                               (norm_differences - (0.5 * threshold))))

            huber_scores = torch.sum(norm_differences_new, dim=1)

            past_dialog_hidden = cur_dialog_hidden
            similarity_scores_mean[round] = similarity_scores_mean[round] + torch.mean(similarity_scores)

            norm_difference_scores_mean[round] = norm_difference_scores_mean[round] + torch.mean(norm_difference_scores)
            norm_scores_mean[round] = norm_scores_mean[round] + torch.mean(norm_scores)
            huber_scores_mean[round] = huber_scores_mean[round] + torch.mean(huber_scores)

            for j in range(batchSize):
                question_str = to_str_pred(questions[j], quesLens[j]) \
                    if qBot is not None else to_str_gt(gtQuestions[j])

                gt_question_str = to_str_pred(gtQuestions[j,round,:], gtQuesLens[j,round])

                gt_questions_str[j].append(gt_question_str[8:])

                question_str_list[j].append(question_str[8:])
                answer_str = to_str_pred(answers[j], ansLens[j])
                if output_dialog:
                    if round == 0:
                        norm_score = float(norm_scores[j])
                        dialog[j]['dialog'].append({
                            "answer": answer_str[8:],
                            "question": question_str[8:] + ":" + "N:%.2f" % norm_score + " "
                        })  # "8:" for indexing out initial <START>
                    else:
                        similarity_score = float(similarity_scores[j])
                        norm_difference_score = float(norm_difference_scores[j])
                        norm_score = float(norm_scores[j])
                        huber_score = float(huber_scores[j])
                        dialog[j]['dialog'].append({
                            "answer": answer_str[8:],
                            "question": question_str[8:] + ":" + "C:%.2f" % similarity_score + ";" +
                                        "NP:%.2f" % norm_difference_score + "H:%.2f" % huber_score + ";" +
                                        "N:%.2f" % norm_score + " "
                        })  # "8:" for indexing out initial <START>
        per_round_bleu_batch = np.zeros((numRounds, batchSize))
        for j in range(batchSize):
            # calculate bleu scores for each question str, with other questions as references to calculate
            # mutual overlap
            # also calculate round by round bleu score
            unigrams = []
            bigrams = []
            avg_bleu_score = 0
            for rnd in range(numRounds):
                # Novel sentences metric
                cur_ques = question_str_list[j][rnd]
                gt_ques = gt_questions_str[j][rnd]
                if cur_ques not in train_questions:
                    novel_questions += 1

                # question oscillation metrics
                if rnd >= 2:
                    if cur_ques == question_str_list[j][rnd-2]:
                        oscillating_questions_cnt += 1

                # bleu/mutual overlap metric
                references = []
                for k in range(numRounds):
                    if rnd != k:
                        references.append(nltk.word_tokenize(question_str_list[j][k]))

                avg_bleu_score += sentence_bleu(references,nltk.word_tokenize(cur_ques))
                per_round_bleu_batch[rnd][j] = sentence_bleu([nltk.word_tokenize(gt_ques)],
                                                             nltk.word_tokenize(cur_ques))
                unigrams.extend(list(ngrams(nltk.word_tokenize(cur_ques),1)))
                bigrams.extend(list(ngrams(nltk.word_tokenize(cur_ques),2)))

            avg_bleu_score /=  float(numRounds)
            mutual_overlap_list.append(avg_bleu_score)
            bleu_metric += avg_bleu_score
            tot_tokens = len(unigrams)

            unigram_ctr = Counter(unigrams)
            bigram_ctr = Counter(bigrams)
            cur_ent_1 = get_entropy_ctr(unigram_ctr)
            ent_1 += cur_ent_1
            ent_1_list.append(cur_ent_1)
            cur_ent_2 = get_entropy_ctr(bigram_ctr)
            ent_2 += cur_ent_2
            ent_2_list.append(cur_ent_2)

            dist_1 = len(unigram_ctr.keys())/float(tot_tokens)
            dist_2 = len(bigram_ctr.keys())/float(tot_tokens)

            dist_1_list.append(dist_1)
            dist_2_list.append(dist_2)

            cur_unique_ques = len(set(question_str_list[j]))
            unique_questions += cur_unique_ques
            unique_questions_list.append(cur_unique_ques)
            # dialog[j]['caption'] += ':' + str(cur_unique_ques)

        tot_examples += batchSize

        if output_dialog:
            text['data'].extend(dialog)

        per_round_bleu += np.sum(per_round_bleu_batch,axis=1)
        avg_precision_list.extend(np.mean(per_round_bleu_batch,axis=0).tolist())

    similarity_scores_mean = similarity_scores_mean * (1.0/tot_idx)
    norm_difference_scores_mean = norm_difference_scores_mean * (1.0/tot_idx)
    norm_scores_mean = norm_scores_mean *(1.0/tot_idx)
    huber_scores_mean = huber_scores_mean *(1.0/tot_idx)

    print("Mean Cos Similarity Scores:", similarity_scores_mean)
    print("Mean Difference of Norms Scores:", norm_difference_scores_mean)
    print("Mean Norm of Dialog State:", norm_scores_mean)
    print("Mean Huber Loss(Norm of differences):", huber_scores_mean)

    text['opts'] = {
        'qbot': params['qstartFrom'],
        'abot': params['startFrom'],
        'backend': 'cudnn',
        'beamLen': 20,
        'beamSize': beamSize,
        'decoder': params['decoder'],
        'encoder': params['encoder'],
        'gpuid': 0,
        'imgNorm': params['imgNorm'],
        'inputImg': params['inputImg'],
        'inputJson': params['inputJson'],
        'inputQues': params['inputQues'],
        'loadPath': 'checkpoints/',
        'maxThreads': 1,
        'resultPath': 'dialog_output/results',
        'sampleWords': 0,
        'temperature': 1,
        'useHistory': True,
        'useIm': True,
    }
    unique_questions_arr = np.array(unique_questions_list)

    # converting metrics to numpy arrays
    similarity_scores_mean = similarity_scores_mean.cpu().data.numpy().tolist()
    norm_difference_scores_mean = norm_difference_scores_mean.cpu().data.numpy().tolist()
    norm_scores_mean = norm_scores_mean.cpu().data.numpy().tolist()
    huber_scores_mean = huber_scores_mean.cpu().data.numpy().tolist()

    bleu_metric /= float(tot_examples)
    ent_1 /= float(tot_examples)
    ent_2 /= float(tot_examples)
    per_round_bleu = per_round_bleu / float(tot_examples)

    print("tot unique questions: ", unique_questions)
    print("tot examples: ", tot_examples)
    print("avg unique questions per example: ", float(unique_questions) / tot_examples)
    print("std unique questions per example: ", float(np.std(unique_questions_arr)))
    print("Mutual Overlap (Bleu Metric): ", bleu_metric)
    print("tot novel questions: ", novel_questions)
    tot_questions = tot_examples * numRounds
    print("tot questions: ", tot_questions)
    print("avg novel questions: ", float(novel_questions)/float(tot_questions))

    print("avg oscillating questions count", float(oscillating_questions_cnt)/tot_questions)
    print("osciallation questions count", oscillating_questions_cnt)

    dataset.split = old_split

    ret_metrics = {}
    ret_metrics["tot_unique_questions"] = unique_questions
    ret_metrics["tot_examples"] = tot_examples
    ret_metrics["mean_unique_questions"] = int((float(unique_questions) / tot_examples) * 100)/100.0
    ret_metrics["std_unique_questions"] =  int(float(np.std(unique_questions_arr)) * 100)/100.0

    ret_metrics["similarity_scores_mean"] = similarity_scores_mean
    ret_metrics["norm_difference_scores_mean"] = norm_difference_scores_mean
    ret_metrics["norm_scores_mean"] = norm_scores_mean
    ret_metrics["huber_scores_mean"] = huber_scores_mean

    ret_metrics["mutual_overlap_score"] = bleu_metric
    ret_metrics["tot_novel_questions"] = novel_questions
    ret_metrics["avg_novel_questions"] = float(novel_questions)/float(tot_questions)
    ret_metrics["tot_questions"] = tot_questions
    ret_metrics['NLL'] = rankMetrics['logProbsMean']

    ret_metrics["average_precision"] = np.mean(per_round_bleu)
    ret_metrics["per_round_precision"] = per_round_bleu.tolist()
    ret_metrics["ent_1"] = ent_1
    ret_metrics["ent_2"] = ent_2
    ret_metrics["dist_1"] = np.mean(dist_1_list)
    ret_metrics["dist_2"] = np.mean(dist_2_list)

    ret_metrics["average_precision_CI"] = (1.96 * np.std(avg_precision_list))/math.sqrt(len(avg_precision_list))
    ret_metrics["ent_1_CI"] = (1.96 * np.std(ent_1_list))/math.sqrt(len(ent_1_list))
    ret_metrics["ent_2_CI"] = (1.96 * np.std(ent_2_list))/math.sqrt(len(ent_2_list))
    ret_metrics["unique_questions_CI"] = (1.96 * np.std(unique_questions_list))/math.sqrt(len(unique_questions_list))
    ret_metrics["mutual_overlap_CI"] = (1.96 * np.std(mutual_overlap_list))/math.sqrt(len(mutual_overlap_list))
    ret_metrics["dist_1_CI"] = (1.96 * np.std(dist_1_list))/math.sqrt(len(dist_1_list))
    ret_metrics["dist_2_CI"] = (1.96 * np.std(dist_2_list))/math.sqrt(len(dist_2_list))

    return text,ret_metrics

def get_entropy_ctr(ctr):

    values = list(ctr.values())
    sum_values = float(sum(values))
    probs = [x/sum_values for x in values]
    return entropy(probs)