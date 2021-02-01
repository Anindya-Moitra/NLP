# Process open ended text using NLP for sentiment analysis, similarity analysis and identifying ‘wh’ questions using NLTK, NumPy, Pandas, and Spacy.

#import dependencies
import pandas as pd
#import statistics
#import numpy as np
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nlp = spacy.load('en_core_web_lg')
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from spacy.tokenizer import Tokenizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2)
nmf_model = NMF(n_components=3, random_state=42)
tokenizer = Tokenizer(nlp.vocab)
sid = SentimentIntensityAnalyzer()
import re
##check lg vector shape
nlp.vocab.vectors.shape


###MAIN FUNCTION START####


def python_clan_analysis():
    """Function to perform battery of programmatic nlp analyses including sentiment, LDA, turn_counts, POS, WC, UNQ"""
    import glob
    #import datetime
    # getting current date and time
    #d = datetime.datetime.today()


    #study_name = "TMWL"
    vs_num = 'VS3'
    cat = "FP_"  # Free play (FP), BookShare(Read), Cleanup(CL)
#     path_input = str(input('Enter path to directory or files--please ensure that they are in the proper format with headings:'))

    path = 'K:\\New Reorganization\\Research\\TMW\\TMW Longitudinal\\CHILDES material\\Transcription\\Completed Transcripts\\CSV_exports\\Session_3\\play\\'
#     path = path_input
    all_files = glob.glob(path + "/*.csv")
    print(all_files)
    c_e_count = 0


    #vs_number = 'VS1'

    li = []


    for filename in all_files:
        try:
            df = pd.read_csv(filename, sep = '|', index_col=None, error_bad_lines=False)
            li.append(df)
        except pd.io.common.EmptyDataError:
            print(filename, " is empty and has been skipped.")
        # df = pd.read_csv(filename, sep = '|', index_col=None, error_bad_lines=False)
        # li.append(df)
        
    #iterate over dfs in list, seperate parent and child speech, sum clean and tokenize data using spacy, similarity anlaysis, 
    #sentiment analsysis, append data sort df by speech onset 

    stop_words = ['b', 'c', 'd', 'e', 'f', 'g', 'h',  'j'
                    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                    'v', 'w', 'x', 'y', 'z']

    #     count = -1


    analyzed_speech_2 = []
    # result = []
    # wh_cols = ['who', 'what', 'when', 'where', 'why', 'how']

    for df in li:
        df['speech'] = df['speech'].apply(str)
        df = df[pd.notnull(df['speech'])]
        df=df.sort_values(by=['speech_onset', 'speech_offset'], ascending=[True, False])
        
       # For 'wh' analysis
        # for col in wh_cols:
        #     if col not in df.columns:
        #         df[col] = 0
    
    
        p_df = df.loc[df['speaker_label'] == 'p_speech']
        c_df = df.loc[df['speaker_label'] == 'c_speech']
        
        if p_df.empty and c_df.empty:
            print("file empty")
            continue
        
    #     p_df= df[p_neg]
    #     c_df= df[c_neg]
        if c_df.empty:
            c_e_count += 1
            print('DataFrame is empty!')
            p_speech = p_df['speech'].sum()
            p_speech = re.split('[?.,]', p_speech)
            p_speech = ' '.join([word.lower() for word in p_speech])
            filtered_parent_speech = p_speech.translate({ord(c): " " for c in ";,:1234567890@#$%^&*()[]{}/<>\|`~-=_+"})
            filtered_parent_speech = re.sub(' +', ' ', filtered_parent_speech).strip().split()
            filtered_parent_speech = ' '.join([word.lower() for word in filtered_parent_speech if word in nlp.vocab and word not in stop_words])
            filtered_parent_speech_unq = ' '.join(set(filtered_parent_speech.split()))
    #         parent_verb_types = check_verb(filtered_parent_speech)
    #         for k, v in parent_verb_types[0].items():
    #             df['Parent_'+ k] = v
    #         parent_speech_pos = pos_counter(filtered_parent_speech)
    #         for k, v in parent_speech_pos.items():
    #             df['Parent_' + k] = v
    
            parent_sentence = sent_complexity(filtered_parent_speech)
            df['PCG_sent_complexity'] = parent_sentence
            parent_speech_pos = pos_counter(filtered_parent_speech)
            #for k, v in parent_speech_pos.items():
                #df['Parent_' + k] = v
                
            df['CHI_sent_complexity'] = 0
 
            
            p_tokens = tokenizer(str(filtered_parent_speech))
            p_tokens_unq = tokenizer(str(filtered_parent_speech_unq))
            df[vs_num + cat + 'CTC'] = 0
            df[vs_num + cat + 'TotUtt'] = len(df)
            #df['child_mlu'] = 0
            df[vs_num + cat + 'PCG_TotUtt'] = len(p_df)
            df[vs_num + cat + 'CHI_TotUtt'] = 0
            df[vs_num + cat + 'PCG_Tokens'] = len(p_tokens)
            df[vs_num + cat + 'PCG_Types'] = len(p_tokens_unq)
                     
            parent_unq_lemmas = []
            for token in nlp(filtered_parent_speech):
                if token.lemma_ not in parent_unq_lemmas:
                    parent_unq_lemmas.append(token.lemma_)
                        
            df[vs_num + cat + 'PCG_Lemmas'] = len(parent_unq_lemmas)
            df[vs_num + cat + 'CHI_Lemmas'] = 0
            df[vs_num + cat + 'CHI_Tokens'] = 0
            df[vs_num + cat + 'CHI_Types'] = 0
            df[vs_num + cat + 'PCG_sentiment']=df['speech'].apply(lambda speech: sid.polarity_scores(speech))

            #df['WH_Questions_Sum'] = result['wh_q'].sum()
    #         df['speech_similarity'] = 0
            df['between_turn_similarity'] = 0
            if len(p_tokens) == 0:
                df[vs_num + cat + 'PCG_TTR'] = len(p_tokens_unq)/1
            else:
                df[vs_num + cat + 'PCG_TTR'] = len(p_tokens_unq)/len(p_tokens)
            #df['lexical_density_parent'] = len(p_tokens_unq)/len(p_tokens)
            df[vs_num + cat + 'CHI_TTR'] = 0
            df[vs_num + cat + 'PCG_MLUtokens'] = len(p_tokens)/len(p_df)
            df[vs_num + cat + 'CHI_MLUtokens'] = 0
            df[vs_num + cat + 'PCG_MLUtypes'] = len(p_tokens_unq)/len(p_df)
            df[vs_num + cat + 'CHI_MLUtypes'] = 0
    #         df['freq_label_words'] = str(label_words)
            name = df['file_name'][0]
            #count += 1
            #print(count)
            print(name)
            print('============================')
    #         df.to_csv(f'{name}_speech_similarity_sentiment_VS4_720.csv')
            analyzed_speech_2.append(df)

        else:
            print('no')
            p_speech = p_df['speech'].sum()
            c_speech = c_df['speech'].sum()
            p_speech = re.split('[.!?]', p_speech)
            c_speech = re.split('[.!?]', c_speech)
            p_speech = ' '.join([word.lower() for word in p_speech])
            c_speech = ' '.join([word.lower() for word in c_speech])
            filtered_parent_speech = p_speech.translate({ord(c): " " for c in ";,:1234567890@#$%^&*()[]{}/<>\|`~-=_+"})
            filtered_child_speech = c_speech.translate({ord(c): " " for c in ";,:1234567890@#$%^&*()[]{}/<>\|`~-=_+"})
            filtered_parent_speech = re.sub(' +', ' ', filtered_parent_speech).strip().split()
            filtered_child_speech = re.sub(' +', ' ', filtered_child_speech).strip().split()
            filtered_parent_speech = ' '.join([word.lower() for word in filtered_parent_speech if word in nlp.vocab and word not in stop_words])
            filtered_child_speech = ' '.join([word.lower() for word in filtered_child_speech if word in nlp.vocab and word not in stop_words])
            # parent_words = Counter(filtered_parent_speech.split())
            # child_words = Counter(filtered_child_speech.split())
            filtered_parent_speech_unq = ' '.join(set(filtered_parent_speech.split()))
            filtered_child_speech_unq = ' '.join(set(filtered_child_speech.split()))
            # p_df['speech'] = p_df['speech'].map(lambda x: str_cleaner(x))
            # c_df['speech'] = c_df['speech'].map(lambda x: str_cleaner(x))
            
            parent_sentence = sent_complexity(filtered_parent_speech)
            df['PCG_sent_complexity'] = parent_sentence
            parent_speech_pos = pos_counter(filtered_parent_speech)
            # for k, v in parent_speech_pos.items():
            #     df['Parent_' + k] = v
                
            child_sentence = sent_complexity(filtered_child_speech)
            df['CHI_sent_complexity'] = child_sentence
            
            
            ## split filtered speech.split()
            # for i in splitlist:
                #result.append(sent_complexity(i))
                # count how many complex, simple, zero
                
                
            # within the same loop call wh_question(i, 'wh column')
            # store the output to some list and save to a dataframe column
            # modify the function to calculate sum of occurance of who, how, etc questions.
                
            
            
            p_tokens = tokenizer(filtered_parent_speech)
            c_tokens = tokenizer(filtered_child_speech)
            p_tokens_unq = tokenizer(str(filtered_parent_speech_unq))
            c_tokens_unq = tokenizer(str(filtered_child_speech_unq))
            df['number_conversational_turns'] = (df['speaker_label'] != df['speaker_label'].shift(axis=0)).sum(axis=0)
            df['ctc'] = (df['speech_onset'] - df['speech_offset'].shift(axis=0) <= 5000).where(df['speaker_label'].shift() != df['speaker_label'])
            df_ctc = df.loc[df['ctc'] == 1].copy().reset_index()
            df_ctc['speech'] = df_ctc.speech.str.replace("[^a-zA-Z0-9 .,']", " ")
            f = df_ctc["speech"] != "   "
            df_filter = df_ctc[f].copy()
            df_filter['speech']=df_filter['speech'].apply(nlp)
            count = 0
            parent_unq_lemmas = []
            for token in nlp(filtered_parent_speech):
                if token.lemma_ not in parent_unq_lemmas:
                    parent_unq_lemmas.append(token.lemma_)
            child_unq_lemmas = []
            for token in nlp(filtered_child_speech):
                if token.lemma_ not in child_unq_lemmas:
                    child_unq_lemmas.append(token.lemma_)
            pl = []
            cl = []
            for i, r in df_filter.iterrows():
                count += 1
                if count % 2 == 0:
                    pl.append(r['speech'])
                else:
                    cl.append(r['speech'])
                    
            # Similarity analysis
            similarity_list = [x.similarity(y) for x, y in zip(cl, pl)]
            if len(similarity_list) == 0:
                df['between_turn_similarity'] = sum(similarity_list)/1
            else:
                df['between_turn_similarity'] = sum(similarity_list)/len(similarity_list)
    
    
    
            df[vs_num + cat + 'CTC'] = df['ctc'].sum()
            df.drop(columns=['ctc'])
            df[vs_num + cat + 'TotUtt'] = len(df)
            df[vs_num + cat + 'PCG_TotUtt'] = len(p_df)
            df[vs_num + cat + 'CHI_TotUtt'] = len(c_df)
            df[vs_num + cat + 'PCG_Tokens'] = len(p_tokens)
            df[vs_num + cat + 'CHI_Tokens'] = len(c_tokens)
            df[vs_num + cat + 'PCG_Types'] = len(p_tokens_unq)
            df[vs_num + cat + 'CHI_Types'] = len(c_tokens_unq)
            #df['PCG_TTR'] = len(p_tokens_unq)/len(p_tokens)
            df[vs_num + cat + 'PCG_Lemmas'] = len(parent_unq_lemmas)
            df[vs_num + cat + 'CHI_Lemmas'] = len(child_unq_lemmas)
            if len(p_tokens) == 0:
                df[vs_num + cat + 'PCG_TTR'] = len(p_tokens_unq)/1
            else:
                df[vs_num + cat + 'PCG_TTR'] = len(p_tokens_unq)/len(p_tokens)
            if len(c_tokens) == 0:   
                df[vs_num + cat + 'CHI_TTR'] = len(c_tokens_unq)/1
            else:
                df[vs_num + cat + 'CHI_TTR'] = len(c_tokens_unq)/len(c_tokens)
            df[vs_num + cat + 'PCG_sentiment']=df['speech'].apply(lambda speech: sid.polarity_scores(speech))
            df[vs_num + cat + 'PCG_MLUtokens'] = len(p_tokens)/len(p_df)
            df[vs_num + cat + 'CHI_MLUtokens'] = len(c_tokens)/len(c_df)
            df[vs_num + cat + 'PCG_MLUtypes'] = len(p_tokens_unq)/len(p_df)
            df[vs_num + cat + 'CHI_MLUtypes'] = len(c_tokens_unq)/len(c_df)
               
            #df['WH_Questions_Sum'] = result['wh_q'].sum()
            
            
            
            
    #             df['parent_words'] = parent_words
    #             df['child_words'] = child_words 
    #             df['label_word_groups'] = label_words
    #         df['freq_label_words'] = str(label_words)
            name = df['file_name'][0]
    #             count += 1
    #             print(count)
    #             print(name)
    #             print(filtered_child_speech_unq)
    #             print(df['speech_similarity'][1])
            print(df['file_name'][0])
            print(df[vs_num + cat + 'CHI_Tokens'][0])
    #             print(parent_words)
    #             print(filtered_child_speech)
    #             print(c_speech)
    #             print(c_tokens)

            print('============================')
    #         df.to_csv(f'{name}_speech_similarity_sentiment_VS4_720.csv')
            del df['ctc']
            analyzed_speech_2.append(df)

            d_list = []


    # for r in analyzed_speech_2:
    #     for n in output_vals:
    #         if n not in r.columns:
    #             r[n] = 0

    d_dict = {}

    for r in analyzed_speech_2:
        for col in r.columns:
            print(col)
            d_dict.update({col: r[col][0]})
        d_list.append(d_dict.copy())

    
    bs_df_1 = pd.DataFrame(d_list)
    print (bs_df_1)
    semantic_sentiment_sum_df = bs_df_1

    return semantic_sentiment_sum_df.to_csv('K:\\New Reorganization\\Research\\TMW\\TMW Longitudinal\\CHILDES material\\Transcription\\Completed Transcripts\\CSV_exports\\VS3_play_NLP_output.csv', index=False)

###START HELPER FUNCTIONS###

def pos_counter(string):
     return dict(Counter([token.tag_ for token in nlp(string)]))

# def str_cleaner(x):
#     x = ' '.join(re.sub(' +', ' ', x.translate({ord(c): " " for c in ";,1234567890@#$%^&*()[]{}/<>\|`~-=_+"})).strip().split())
#     return x


def sent_complexity(x):
    verb_count = 0
    x = nlp(x)
    if len(x) >= 2:
        for token in x:
            if token.pos_ == 'VERB':
                verb_count += 1
                continue
        if verb_count == 0:
            return "zero complexity sentence"
        elif verb_count == 1:
            return "simple-complexity sentence"
        else: 
            return "complex sentence"
        
    else:
        return "one word utt"

# def wh_questions(doc):
#     wh_questions = ['what', 'why', 'how', 'when','where', 'who']
#     wh_nlp = []
#     wh_dict = {}
#     for q in wh_questions:
#         wh_nlp.append(nlp(q))
#     token_list = []
#     doc = nlp(doc)
#     for token in doc:
#         token_list.append(token)        
#     for i, token in enumerate(token_list):
#         if str(token) in wh_questions and str(token_list[-1]) == '?':
#             wh_dict[str(token_list[i])] = 1
#             return wh_dict
#     else:
#         wh_dict['Null'] = 0
#         return wh_dict


# def wh_question(doc, wh_q, whq2=''):
#     token_list = []
#     doc = nlp(doc)
#     for token in doc:
#         token_list.append(token)        
#     if str(token_list[-1]) == '?':
#         for token in token_list:
#             if str(token) == wh_q:
#                 return 1
#         else:
#              return 0
    
#     else:
#         return 0

def zero_sent_complexity(x):
    verb_count = 0
    x = nlp(x)
    if len(x) >= 2:
        for token in x:
            if token.pos_ == 'VERB':
                verb_count += 1
                continue
        if verb_count == 0:
            return 1
        else:
            return 0


def simple_sent_complexity(x):
    verb_count = 0
    x = nlp(x)
    if len(x) >= 2:
        for token in x:
            if token.pos_ == 'VERB':
                verb_count += 1
                continue
        if verb_count == 1:
            return 1
        else:
            return 0

def complex_sent_complexity(x):
    verb_count = 0
    x = nlp(x)
    if len(x) >= 2:
        for token in x:
            if token.pos_ == 'VERB':
                verb_count += 1
                continue
        if verb_count > 1:
            return 1
        else:
            return 0


# def one_word_question(x):
#     if type(x) == str:
#         x = nlp(x)
#     else:
#         x=x
#     sent_els = []
#     for token in x:
#         sent_els.append(str(token))
#     if len(sent_els) == 2 and '?' in sent_els:
#         return 1
#     else:
#         return 0


# ###END HELPER FUNCTIONS###


if __name__ == "__main__":
    python_clan_analysis()


