import os
import numpy as np
import tensorflow as tf
import random

from argparse import ArgumentParser

from athene.retrieval.document.doc_retrieval import Doc_Retrieval
from athene.retrieval.sentences.data_processing.data import Data
from athene.retrieval.sentences.data_processing.sentence_loader import SentenceDataLoader
from athene.retrieval.sentences.deep_models.ESIM import ESIM as SentenceESIM
from athene.retrieval.sentences.ensemble import scores_processing, post_processing, prediction_processing, \
    prediction_processing_no_reload, training_phase
from athene.retrieval.sentences.ensemble import entrance as sentence_retrieval_ensemble_entrance
from athene.utils.config import Config

parser = ArgumentParser()
parser.add_argument("--db-path",required=True)
parser.add_argument("--random-seed",default=1234)
parser.add_argument("--sentence-model", required=True)
parser.add_argument("--c-max-length", default=20)
parser.add_argument("--s-max-length", default=60)
parser.add_argument("--fasttext-path", default="data/fasttext/wiki.en.bin")
parser.add_argument("--train-data", default="data/fever/train.wiki7.jsonl")
parser.add_argument("--dev-data", default="data/fever/dev.wiki7.jsonl")
parser.add_argument("--test-data", default="data/fever/test.wiki7.jsonl")
parser.add_argument("--add-claim", default=True)

args = parser.parse_args()

k_wiki = 7

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_iwords(prog_args, retrieval):
    print("Getting iwords")
    random.seed(prog_args.random_seed)

    args = Config.sentence_retrieval_ensemble_param
    args.update(vars(prog_args))
    
    args = Struct(**args)

    print(args.train_data)
    print(args.dev_data)
    print(args.test_data)

    data = Data(args.sentence_model, args.train_data, args.dev_data, args.test_data, args.fasttext_path,
                num_negatives=args.num_negatives, h_max_length=args.c_max_length, s_max_length=args.s_max_length,
                random_seed=args.random_seed, reserve_embed=args.reserve_embed, db_filepath=args.db_path, load_instances=False, retrieval=retrieval)

    return data.word_dict, data.iword_dict



def setup():
    # Set seeds
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Document Retrieval
    retrieval = Doc_Retrieval(database_path=args.db_path, add_claim=args.add_claim, k_wiki_results=k_wiki)

    # Sentence Selection
    words, iwords = get_iwords(args, retrieval)
    sentence_loader = SentenceDataLoader(fasttext_path=args.fasttext_path, db_filepath=args.db_path, h_max_length=args.c_max_length, s_max_length=args.s_max_length, reserve_embed=True)
    sentence_loader.load_models(words,iwords)

    sargs = Config.sentence_retrieval_ensemble_param
    sargs.update(args)
    sargs = Struct(**sargs)
    selection = SentenceESIM(h_max_length=sargs.c_max_length, s_max_length=sargs.s_max_length, learning_rate=sargs.learning_rate,
                       batch_size=sargs.batch_size, num_epoch=sargs.num_epoch, model_store_dir=sargs.sentence_model,
                       embedding=sentence_loader.embed, word_dict=sentence_loader.iword_dict, dropout_rate=sargs.dropout_rate,
                       num_units=sargs.num_lstm_units, share_rnn=sargs.share_parameters, activation=tf.nn.tanh)

    def get_docs_line(line):
        nps, wiki_results, pages = retrieval.exact_match(line)
        line['noun_phrases'] = nps
        line['predicted_pages'] = pages
        line['wiki_results'] = wiki_results
        return line

    def get_docs(lines):
        return list(map(get_docs_line, lines))


    def get_sents(lines):
        indexes, location_indexes = sentence_loader.get_indexes(lines)
        all_predictions = []

        for i in range(args.num_model):
            model_store_path = os.path.join(args.model_path, "model{}".format(i + 1))
            if not os.path.exists(model_store_path):
                raise Exception("model must be trained before testing")

            selection.restore_model(os.path.join(model_store_path, "best_model.ckpt"))
            predictions = []

            for test_index in indexes:
                prediction = selection.predict(test_index)
                predictions.append(prediction)

            all_predictions.append(predictions)
            tf.reset_default_graph()

        ensembled_predicitons = scores_processing(all_predictions, args)
        processed_predictions, scores = post_processing(ensembled_predicitons, location_indexes)
        final_predictions = prediction_processing_no_reload(lines, processed_predictions)

        return final_predictions

    def process_claim(claims):
        claims = get_docs(claims)
        claims = get_sents(claims)
        return claims

    return process_claim([{"claim":"This is a test"}])


setup()
