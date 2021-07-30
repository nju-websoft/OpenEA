import pandas as pd

golden_path = "dataset//book_golden.txt"
predict_path = "result//majority_voting.txt"

def str_to_set(str):
    return set(str.lower().replace(';',' ').replace(',',' ').replace('.',' ').replace(':',' ').replace('&',' ').
               replace('/',' ').replace('\'',' ').replace('(author)',' ').replace('(joint author)',' ').split())

def sim_jaccard (str1,str2) :
    set1 = str_to_set(str1)
    set2 = str_to_set(str2)
    return len(set1&set2)/len(set1|set2)

def measure_result(df_predict,df_answer,answer_col='author'):
    ans_dict = {'0.0-0.2':0,'0.2-0.4':0,'0.4-0.6':0,'0.6-0.8':0,'0.8-1.0':0}
    miss_count = 0
    measure_sum = 0
    measure_hit = 0
    for index,row in df_predict.iterrows():
        if index not in df_answer.index:
            miss_count = miss_count + 1
        else:
            str1 = row[answer_col]
            str2 = df_answer.loc[index,answer_col]
            simmality = sim_jaccard(str1,str2)
            if simmality>=0.8:
                measure_hit+=1
                ans_dict['0.8-1.0']+=1
            elif simmality>=0.6:
                ans_dict['0.6-0.8']+=1
            elif simmality>=0.4:
                ans_dict['0.4-0.6']+=1
            elif simmality>=0.2:
                ans_dict['0.2-0.4']+=1
            else:
                ans_dict['0.0-0.2']+=1
            measure_sum += simmality
    print('miss_count',miss_count)
    print('measure_sum',measure_sum)
    print('measure_hit',measure_hit)
    print('answer_dict',str(ans_dict))

df_predict = pd.read_csv(predict_path,sep='\t',index_col='isbn')
df_answer = pd.read_csv(golden_path,sep='\t',names=['isbn','author'],index_col='isbn')
measure_result(df_predict,df_answer)